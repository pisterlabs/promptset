#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFaceHub, OpenAI, CTransformers
from langchain.chat_models import ChatOpenAI
import os
import argparse
import time
from chromadb.config import Settings

import pandas as pd
from pandasai import PandasAI
from constants import llms, openai_index
# Instantiate a LLM
#from pandasai.llm.openai import OpenAI

#from processDocument import get_list_sources
load_dotenv()


#model_path_gpt4all = os.environ.get('GPT4ALL_PATH')
#model_path_llamacpp = os.environ.get('LLAMACPP_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
models_path = os.environ.get('MODELS_PATH')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def ask_question(project_code, query, llm_name, embeddings_model_name, main_model, api_key, selected_docs, mute_stream=False, hide_source=False, chat_history=[]):
    persist_directory = os.environ.get('SOURCE_DBS', 'source_dbs') + "/" +  project_code
    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
    )
    # Create embeddings
    if llm_name == llms[openai_index]:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embeddings_model_name)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    #retriever = db.as_retriever()

    if len(selected_docs) > 1:
        list_filter = []
        for sc in selected_docs:
            list_filter.append({'source': {'$eq': sc}})
        search_kwargs = {"k": target_source_chunks, "filter":{'$or': list_filter}}
        retriever = db.as_retriever(search_kwargs=search_kwargs)
    elif len(selected_docs) == 1:
        print(selected_docs[0])
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks, 'filter': {'source':selected_docs[0]}})
        #retriever = db.as_retriever(search_kwargs={'source': {'$eq': selected_docs[0]}})
        
    #search_kwargs={"filter":{'$or': [{'source': {'$eq': './SampleDoc/Bikes.pdf'}}, {'source': {'$eq': './SampleDoc/IceCreams.pdf'}}]}}
    #search_kwargs={"filter":{'source': {'$in': ['./SampleDoc/Bikes.pdf']}}}
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    if llm_name == llms[0]:
        llm = HuggingFaceHub(repo_id=main_model, model_kwargs={"temperature":0.5, "max_length":512})
    elif llm_name == llms[1]:
        llm = GPT4All(model=models_path+main_model, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif llm_name == llms[2]:
        llm = LlamaCpp(model_path=models_path+main_model, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif llm_name == llms[3]:
        llm = CTransformers(model=models_path+main_model,model_type="llama", config={'max_new_tokens':128,'temperature':0.01})
    elif llm_name == llms[openai_index]:
        #llm = OpenAI(openai_api_key=api_key)
        if "gpt" not in main_model:
            llm = OpenAI(openai_api_key=api_key,model=main_model)
        else:
            llm = ChatOpenAI(openai_api_key=api_key,model=main_model)
    else:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {llm_name} is not supported. Please choose one of the following: HuggingFaceHub, CTransformers, OpenAI, LlamaCpp, GPT4All")

    all_csv = True
    #list_sources = get_list_sources(project_code, llm_name, embeddings_model_name, api_key)
    for file_source in selected_docs:
        if ".csv" not in file_source:
            all_csv = False

    if all_csv:
        pandas_ai = PandasAI(llm)
        dfs = []
        for file_source in selected_docs:
            df = pd.read_csv(file_source)
            dfs.append(df)
        result = pandas_ai(dfs, prompt=query)
        answer, docs, chat_history = result.__str__(), [], []
    elif (llm_name == llms[openai_index] and "gpt" in main_model) or (llm_name == llms[3]):
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, return_source_documents= not hide_source
        )
        res = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, res["answer"]))
        #p573476
        #tabulate the first 10 HIGHEST energy consumption for Africa between 1980 AND 2000. Include the code, year and energy consumption.
        answer, docs, chat_history = res['answer'].__str__(), [] if hide_source else res['source_documents'], chat_history
    else:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not hide_source)
        res = qa(query)
        answer, docs, chat_history = res['result'].__str__(), [] if hide_source else res['source_documents'], []
    
    return answer, docs, chat_history 