import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH = "models/ggml-gpt4all-j-v1.3-groovy.bin"


def process_file(file):

    # Load PDF file using PyPDFLoader
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    # load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # load LLM model
    llm = GPT4All(model=MODEL_PATH)

    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
                                    llm=llm, 
                                   retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                   return_source_documents=True)
    return chain

def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain
    
    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1
    
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''
