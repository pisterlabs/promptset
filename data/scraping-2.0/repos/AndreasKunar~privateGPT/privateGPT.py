#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain import PromptTemplate
import langchain
import os

load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY','db')
model_type = os.environ.get('MODEL_TYPE','GPT4All')
model_path = os.environ.get('MODEL_PATH','./models/ggml-gpt4all-j-v1.3-groovy.bin')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME','all-MiniLM-L6-v2')
model_n_ctx = os.environ.get('MODEL_N_CTX',5000)
model_temp = os.environ.get('MODEL_TEMP',0.4)
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
mute_stream = os.environ.get('MUTE_STREAM',"False") != "False"
hide_source = os.environ.get('HIDE_SOURCE',"False") != "False"
hide_source_details = os.environ.get('HIDE_SOURCE_DETAILS',"False") != "False"
#langchain debugging
if os.environ.get('LANGCHAIN_DEBUG',"False") != "False":
    langchain.debug=True

from constants import CHROMA_SETTINGS

# gpt4all tweak - use local response streaming callback 
#   ... directly from low-level GPT4All.py,
# define C callback function signatures
import ctypes
GPT4allResponseCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32, ctypes.c_char_p)
# local GPT4all response-streaming callback
def gpt4all_local_callback(token_id, response):
    # gracefully handle utf-8 decoding issues
    print(response.decode('utf-8', errors='ignore'))
    return True

def main():
    print("privateGPT: A (tweaked) private GPT-3 alternative for question answering")
    # moved all command line arguments to .env
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=[StreamingStdOutCallbackHandler()], 
                           temperature=model_temp,n_threads = 4,n_gpu_layers=1) # type: ignore
            # tweak - set verbose=False in underlying LlamaCpp.py to surpress llama_print_timings
            llm.client.verbose = False
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=[], temp=model_temp) # type: ignore
            # tweak - insert callback into low-level GPT4All.py
            llm.client.model._response_callback = GPT4allResponseCallback(gpt4all_local_callback)
        case _default:
            print(f"Model {model_type} not supported!")
            exit()
            
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not hide_source,
        chain_type_kwargs=chain_type_kwargs,
    )
    # Interactive questions and answers
    while True:
        query = input("\n### Enter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain and stream-print the result
        print("\n### Answer by privateGPT via "+os.path.split(model_path)[1]+" with temperature:"+str(model_temp)+":")

        res = qa(query)
        answer, docs = res['result'], [] if hide_source else res['source_documents']

        # Print the relevant sources used for the answer
        if docs:
            print("--- Sources: ----")
        for document in docs:
            print(">>" + os.path.split(document.metadata["source"])[1]+":")
            if not hide_source_details:
                print(document.page_content)
        print("-------------------")

if __name__ == "__main__":
    main()
