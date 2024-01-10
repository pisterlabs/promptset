import os
import speech_recognition as sr
import whisper
import time
import threading
import pyttsx3

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

 
loader = PyPDFLoader("C:\\TallerIA\\taller-bcp-ia\\Memoria.pdf")
documents = loader.load_and_split()

print('Tamaño de los documentos')
print(len(documents))

print(documents[0].page_content)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(documents)

print('Tamaño de texts')
len(texts)


print(texts[0].page_content)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings, persist_directory="db")

model_n_ctx = 1000
model_path = "C:\\Users\\nickq\\AppData\\Roaming\\nomic.ai\\ggml-gpt4all-j-v1.3-groovy.bin"
model = GPT4All(model=model_path, n_ctx=1000, backend="gptj", verbose=False)

qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    verbose=False,
)
     

res = qa(
    "How was the Banco de Credito del Peru established?"
)

print(res)
print(res["result"])
