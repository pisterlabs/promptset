import yaml

from chromadb.config import Settings
from langchain.callbacks.manager import CallbackManager

from langchain.llms import LlamaCpp
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager

callbacks = BaseCallbackManager([StdOutCallbackHandler()])

# Load DVC parameters
params = yaml.safe_load(open("params.yaml"))["inference"]

embeddings_model_name = params["EMBEDDINGS_MODEL_NAME"]
persist_directory = params['PERSIST_DIRECTORY']
source_directory = params['SOURCE_DIRECTORY']
model_type = params['MODEL_TYPE']
model_path  = params['MODEL_PATH']
model_n_ctx = params['MODEL_N_CTX']
chunk_size = params['CHUNK_SIZE']
chunk_overlap = params['CHUNK_OVERLAP']

CHROMA_SETTINGS = Settings(
        chroma_db_impl = params['CHROMA_DB_IMPL'],
        persist_directory = persist_directory,
        anonymized_telemetry = params['ANONYMIZED_TELEMETRY'],
)


def main(): 

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Prepare the LLM
    llm = LlamaCpp(model_path="./model/llama-2-70b.Q5_K_M.gguf", 
               n_threads=12,
               n_parts=-1,
               n_gpu_layers=1, 
               n_batch=512, 
               callback_manager=callbacks, 
               verbose=True,
               max_tokens=10000, 
               n_ctx=8192)
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Interactive questions and answers    
    query = input("\n What can I help you today: ")

    # Get the answer from the chain
    res = qa(query)
    answer, docs = res['result'], res['source_documents']

    # Print the result
    print("\n\n> Instruction(s):")
    print(query)
    print("\n> Answer(s):")
    print(answer)

    # Print the relevant sources used for the answer
    for document in docs:
    
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)


if __name__ == "__main__":

    main()