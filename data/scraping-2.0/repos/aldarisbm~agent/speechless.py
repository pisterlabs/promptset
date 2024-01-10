import json
import os
from dotenv import load_dotenv
from langchain import Prompt

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


load_dotenv()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name = os.getenv("MODEL_PATH")
prompt_file = "./prompt_samples/pf_json_extract"
grammar_file = os.getenv("GRAMMAR_FILE")

llm = LlamaCpp(
    model_path=model_name,
    temperature=0,
    use_mlock=True,
    grammar_path=grammar_file,
    n_batch=1024,
    repeat_penalty=1.1,
    n_ctx=4096,
    n_threads=10,
    n_gpu_layers=30,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

with open(prompt_file) as prompt:
    pr = prompt.read()
    json_result = llm(prompt=pr)
    res = json_result.strip()
    res = json.loads(res)
    print(res)

# pdf_file = "./doc_samples/A-Game-Of-Thrones-Book.pdf"
# loader = PyPDFLoader(pdf_file)
