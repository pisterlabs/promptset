#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS
from langchain.llms import GPT4All, LlamaCpp
from langchain import PromptTemplate
from langchain.model_laboratory import ModelLaboratory
from langchain import debug as LCdebug
import os

# get parameters from .env
from dotenv import load_dotenv
load_dotenv()

# this is modified to use/compare LLM outputs - ignores .env model settings
#   does no streaming or showing of sources
llm_list = [
    {'model_type': 'GPT4All',  'model_path': './models/ggml-gpt4all-j-v1.3-groovy.bin'},
    {'model_type': 'LlamaCpp', 'model_path': './models/koala-13B.ggmlv3.q4_0.bin'}
]
persist_directory = os.environ.get('PERSIST_DIRECTORY','db')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME','all-MiniLM-L6-v2')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
model_n_ctx = os.environ.get('MODEL_N_CTX',5000)
model_temp = os.environ.get('MODEL_TEMP',0.4)
LCdebug=os.environ.get('LANGCHAIN_DEBUG',"False") != "False"

# gpt4all tweak - use local response streaming callback 
#   ... directly from low-level GPT4All.py,
# define C callback function signatures
import ctypes
GPT4allResponseCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32, ctypes.c_char_p)
# local GPT4all response-streaming callback
def gpt4all_local_callback(token_id, response):
    print(response.decode('utf-8', errors='ignore'))
    return True

def main():
    # Banner
    print("privateGPTcompare: A Private Knowledge-Base Answering LLM, comparing multiple LLMs answers")

    # setup retriever
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
   
    # Prepare the LLMs
    llms=[]
    for llm_item in llm_list:        
        match llm_item.get('model_type'):
            case "LlamaCpp":
                llm = LlamaCpp(model_path=llm_item.get('model_path'), n_ctx=model_n_ctx, callbacks=[], temperature=model_temp) # type: ignore
                # tweak - set verbose=False in underlying LlamaCpp.py to surpress llama_print_timings
                llm.client.verbose = False
            case "GPT4All":
                llm = GPT4All(model=llm_item.get('model_path'), n_ctx=model_n_ctx, backend='gptj', callbacks=[], temp=model_temp) # type: ignore
                # tweak - insert callback into low-level GPT4All.py
                llm.client.model._response_callback = GPT4allResponseCallback(gpt4all_local_callback)
            case _default:
                print(f"Model {llm_item.get('model_type')} not supported!")
                exit()
        llms.append(llm) 
    
    # Prepare the LLM prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    
    # setup LLMs-compare with prompt
    model_lab = ModelLaboratory.from_llms(llms)


    # Interactive questions and answers
    while True:
        query = input("\n### Enter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain and stream-print the result
        print("\n### Generating answers ... (temperature="+str(model_temp)+")")

        # get embeddings for query and generate context
        docs=(db.similarity_search(query, n_results = target_source_chunks))
        context=""
        for doc in docs:
            context += doc.page_content + "\n"
   
        # get answers from LLMs
        model_lab.compare(prompt_template.format(context=context, question=query))

        # res = qa(query)
        # answer, docs = res['result'], [] if hide_source else res['source_documents']
      
if __name__ == "__main__":
    main()
