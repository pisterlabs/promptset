"""Ask a question to the notion database."""


import modal
from globals import Globals

stub = modal.Stub(image=modal.Image.debian_slim().pip_install("openai","langchain","faiss-cpu","requests", "cohere"), name="treehacks-server")

@stub.function(secret=modal.Secret.from_name(Globals["OPENAPI_SECRET"]), memory=Globals["MEMORY"], cpu=Globals["CORES"], shared_volumes={Globals["CACHE_DIR"]: Globals["VOLUME"]}, rate_limit=modal.RateLimit(per_minute=5))
def gen_cards(question: str):
    import faiss
    from langchain import OpenAI, Cohere
    from langchain.chains import VectorDBQAWithSourcesChain
    import pickle
    
    # Load the LangChain.
    index = faiss.read_index(Globals["INDEX"])
    
    with open(Globals["FAISS_PKL"], "rb") as f:
        store = pickle.load(f)
    
    store.index = index
    # chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    chain = VectorDBQAWithSourcesChain.from_llm(llm=Cohere(model="gptd-instruct-tft", temperature=0), vectorstore=store)
    prompt = f"Generate flashcards about the topic {question} in a JSON array, where each array element consists of a term (string) and definition (string)."
    result = chain({"question": prompt})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    return result


@stub.webhook
async def predict(question: str):
    return gen_cards.call(question)
