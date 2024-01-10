from modal import Image, gpu, method, Volume, Mount, Secret, Stub, asgi_app
import subprocess
import os
from common import stub, BASE_MODELS, VOLUME_CONFIG
import re

from pathlib import Path

tgi_image = (
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.0.3")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("text-generation")
    .pip_install("transformers>=4.33.0")
    .pip_install("langchain")
    .pip_install("pinecone-client")
    .pip_install("python-dotenv")
    .pip_install("openai")
    .pip_install("tiktoken")
    .pip_install('ratelimit')
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained"))
)


@stub.function(image=tgi_image, volumes=VOLUME_CONFIG, timeout=60 * 20)
def merge(run_id: str, commit: bool = False):
    from text_generation_server.utils.peft import download_and_unload_peft

    os.mkdir(f"/results/{run_id}/merged")
    subprocess.call(f"cp /results/{run_id}/*.* /results/{run_id}/merged", shell=True)

    print(f"Merging weights for fine-tuned {run_id=}.")
    download_and_unload_peft(f"/results/{run_id}/merged", None, False)

    if commit:
        print("Committing merged model permanently (can take a few minutes).")
        stub.results_volume.commit()


@stub.cls(
    image=tgi_image,
    gpu=gpu.A100(count=1, memory=80),
    allow_concurrent_inputs=1000,
    volumes=VOLUME_CONFIG,
    secret=Secret.from_name("modal-openai")
)
class Model:
    def __enter__(self):
        from text_generation import AsyncClient
        import socket
        import time
        import os

        run_id = "chat13-dochelper"

        model = f"/results/{run_id}/merged"


        if run_id and not os.path.isdir(model):
            merge.local(run_id)  # local = run in the same container


        launch_cmd = ["text-generation-launcher", "--model-id", model, "--port", "8000"]
        self.launcher = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL)

        self.client = None
        while not self.client and self.launcher.returncode is None:
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(1.0)
        self.template = """<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user}[/INST] """

        assert self.launcher.returncode is None
    def __init__(self, base: str = "base13", run_id: str = "chat13-dochelper"):
        from text_generation import AsyncClient
        import socket
        import time
        import os

        model = f"/results/{run_id}/merged" if run_id else BASE_MODELS[base]

        if run_id and not os.path.isdir(model):
            merge.local(run_id)  # local = run in the same containe

        print(f"Loading {model} into GPU ... ")
        launch_cmd = ["text-generation-launcher", "--model-id", model, "--port", "8000"]
        self.launcher = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL)
        self.template = """<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user}[/INST] """
        self.client = None
        while not self.client and self.launcher.returncode is None:
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(1.0)

        assert self.launcher.returncode is None

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.launcher.terminate()

    @method()
    async def generate(self, user_msg: str):
        qa_model = load_retrieval.local()
        system_msg = "You are a documentation assistant that finds the search keyword of a user question that can be used to search the documentation. List only the search keyword that you would use to earch, nothing else. The format should of the response Answer: topic\n" + "Here is an example:\nQuestion: How do I upload a file to a volume?\nAnswer: modal volume modal volume put\nQuestion: "
        prompt = self.template.format(system=system_msg, user=user_msg)
        answer = await self.client.generate(prompt, max_new_tokens=512)
        topic = answer.generated_text
        return retrieve_docs.local(qa_model, remove_answer_prefix(topic))
    
    @method()
    async def generate_stream(self, question: str):
        qa_model = load_retrieval.local()
        system_msg = "You are a documentation assistant that finds the search keyword of a user question that can be used to search the documentation. List only the search keyword that you would use to earch, nothing else. The format should of the response Answer: topic\n" + "Here is an example:\nQuestion: How do I upload a file to a volume?\nAnswer: modal volume modal volume put\nQuestion: "
        prompt = self.template.format(system=system_msg, user=question)

        results = []  # Create an empty list to store the yielded values

        async for response in self.client.generate_stream(prompt, max_new_tokens=1024):
            if not response.token.special:
                results.append(response.token.text)  # Append the yielded value to the list

        yield retrieve_docs.local(qa_model, remove_answer_prefix("".join(results)))  # Run retrieve_docs on the joined results


@stub.function(image=tgi_image, volumes=VOLUME_CONFIG, secret=Secret.from_name("modal-openai"))
def load_retrieval():
    from langchain.vectorstores import Pinecone
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    import pinecone
    import dotenv
    import os


# initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
        )
    # select which embeddings we want to use

    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Pinecone.from_existing_index('modal', embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model = "gpt-4", streaming = True), chain_type="stuff", retriever=retriever, return_source_documents=False)
    return qa

@stub.function(image=tgi_image, volumes=VOLUME_CONFIG)
def retrieve_docs(model, query):
    prepend_query = "You are a documentation assistant that will be given a keyword to search through the documentation. Use that keyword to search the documentation and use the results to formulate an answer. Here is the query: "
    ans = model({"query": prepend_query + query})
    return ans["result"]

def remove_answer_prefix(s):
    # Find the line containing 'Answer:'
    answer_line = next((line for line in s.split('\n') if 'Answer:' in line), None)
    if answer_line is None:
        return s
    # Remove the 'Answer:' prefix
    return re.sub(r'^Answer: ', '', answer_line)


# ## Serve the model
# Once we deploy this model with `modal deploy text_generation_inference.py`, we can serve it
# behind an ASGI app front-end. The front-end code (a single file of Alpine.js) is available
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html).
#
# You can try our deployment [here](https://modal-labs--tgi-app.modal.run).

frontend_path = Path(__file__).parent / "llm-frontend"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    keep_warm=1,
    allow_concurrent_inputs=1000,
    timeout=60 * 10,
    image = tgi_image,
    volumes=VOLUME_CONFIG
)
@asgi_app(label="tgi-app")
def app():
    import json
    import ratelimit

    import fastapi
    import fastapi.staticfiles
    from fastapi.responses import StreamingResponse

    web_app = fastapi.FastAPI()

    @ratelimit.limits(calls=1, period=30)
    @web_app.get("/stats")
    async def stats():
        stats = await Model().generate_stream.get_current_stats.aio()
        return {
            "backlog": stats.backlog,
            "num_total_runners": stats.num_total_runners,
        }


# Define the rate limit decorator
    @ratelimit.limits(calls=1, period=60)
    @web_app.get("/completion/{question}")
    async def completion(question: str):
        from urllib.parse import unquote

        async def generate():
            async for text in Model().generate_stream.remote_gen.aio(
                unquote(question)
            ):
                yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app


# ## Invoke the model from other apps
# Once the model is deployed, we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-tgi-Llama-2-70b-chat-hf", "Model.generate")
# >>> f.remote("What is the story about the fox and grapes?")
# 'The story about the fox and grapes ...
# ```

@stub.local_entrypoint()
def main(prompt: str):
    from dotenv import load_dotenv
    load_dotenv()
    print(f"Running completion for prompt:\n{prompt}")
    qa_model = load_retrieval.local()

    system_msg = "You are a documentation assistant that finds the search keyword of a user question that can be used to search the documentation. List only the search keyword that you would use to earch, nothing else. The format should of the response Answer: topic\n" + "Here is an example:\nQuestion: How do I upload a file to a volume?\nAnswer: modal volume modal volume put\nQuestion: "

    user_msg = prompt

    #print("=" * 20 + "Generating without adapter" + "=" * 20)
    #for output in Model(base).generate.map([prompt] * batch):
    #    print(output)

    answer = Model().generate.remote(prompt)
    print("ANSWER: ")
    print(answer)