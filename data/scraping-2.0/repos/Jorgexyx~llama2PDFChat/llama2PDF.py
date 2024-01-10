from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index import download_loader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, LangchainEmbedding, VectorStoreIndex
from llama_index.llms import LangChainLLM

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_path = "INSERT MODEL PATH FROM ROOT"
llama_llm=LlamaCpp(
    model_path=model_path,
    input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048
)

llm = LangChainLLM(llm=llama_llm)

embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(llm=llm,  embed_model=embed_model)

def index_file(file_name):
    print ("Loading file");
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(file_name))

    return VectorStoreIndex.from_documents(documents, service_context=service_context)

def chat(index):
    while True:
        prompt = input("\nPrompt: ")
        query_engine = index.as_query_engine(response_mode="compact")

        response = query_engine.query(prompt)
        print("\nResponse: " + str(response))

file_name = ""
index = index_file(file_name)
chat(index)
