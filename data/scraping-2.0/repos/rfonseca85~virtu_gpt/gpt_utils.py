#!/usr/bin/env python3
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import time

from constants import CHROMA_SETTINGS

qa_model = None


def load_model(embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch,
               target_source_chunks, mute_stream):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS,
                client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                           verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                          callbacks=callbacks, verbose=False)
        case _default:
            raise Exception(
                f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

############################################## SIDEBAR ##############################################

if not load_dotenv():
    st.error("Could not load .env file or it is empty. Please check if it exists and is readable.")

embeddings_model_name = st.sidebar.text_input("Embeddings Model Name", os.environ.get("EMBEDDINGS_MODEL_NAME"))
persist_directory = st.sidebar.text_input('Persist Directory', os.environ.get('PERSIST_DIRECTORY'))

model_type = st.sidebar.selectbox('Model Type', ['LlamaCpp', 'GPT4All'])
model_path = st.sidebar.text_input('Model Path', os.environ.get('MODEL_PATH'))
model_n_ctx = st.sidebar.number_input('Model Context (N_CTX)', value=int(os.environ.get('MODEL_N_CTX', 0)))
model_n_batch = st.sidebar.number_input('Model Batch Size (N_BATCH)', value=int(os.environ.get('MODEL_N_BATCH', 8)))
target_source_chunks = st.sidebar.number_input('Target Source Chunks',
                                               value=int(os.environ.get('TARGET_SOURCE_CHUNKS', 4)))
hide_source = st.sidebar.checkbox('Hide Source Documents')
mute_stream = st.sidebar.checkbox('Mute Stream')

load_button = st.sidebar.button('Load Model')

if load_button:
    with st.spinner("Loading model..."):
        qa_model = load_model(embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx,
                              model_n_batch, target_source_chunks, mute_stream)
        st.success('Model loaded successfully!')

############################################## PAGE ##############################################
st.title("Interactive LLM Interface")

query = st.text_input("Enter a query:", placeholder="e.g. What is the capital of Germany?")
if query:
    with st.spinner("Processing..."):
        start = time.time()
        res = qa_model(query)
        answer = res.get('result', 'No answer found')
        docs = res.get('source_documents', []) if not hide_source else []
        # answer, docs = res['result'], [] if hide_source else res['source_documents']
        end = time.time()

        st.write("### Question:")
        st.write(query)
        st.write(f"### Answer (took {round(end - start, 2)} s.):")
        st.write(answer)

        if not hide_source:
            for document in docs:
                st.write("#### Source Document:")
                st.write(document.metadata["source"])
                st.write(document.page_content)


