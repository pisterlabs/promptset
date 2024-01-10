__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

from langchain import hub
obj = hub.pull("rlm/rag-prompt-mistral")


# Load the embedding function
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Initialize Vector Store and use it as retriever
db = Chroma(
    "oscar_docs", embedding_function=embedding_function, persist_directory="db"
)
retriever = db.as_retriever()

template = """Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 100  # Change this value based on your model and your GPU VRAM pool.
n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 4096 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="../llama.cpp/models/mistral-7b-instruct-v0.1.Q8_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=n_ctx,
    callback_manager=callback_manager,
    max_tokens=4096,
    verbose=True, # Verbose is required to pass to the callback manager
    temperature=0.2,
)

retrieval_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    retriever=retriever,
    llm=llm,
    return_source_documents=True,
    chain_type_kwargs={"prompt": obj},
)
question = "How to use Ampere architecture GPUs with CUDA support?"

retrieval_chain({"query": question})


