from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp

# load some Tender
from TendersWA.Preprocessing.Tender import Tender
import os
os.chdir("/home/ucc/maxichat/Capstone/data/tender_raw")
t = Tender.load("ZPA980122.pickle")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import EmbeddingsRedundantFilter
from langchain.schema.document import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
documents = text_splitter.split_documents([Document(page_content = text) for text in t.file_map.values()])

llamaEmbeddings = LlamaCppEmbeddings(
    model_path="/home/ucc/maxichat/Capstone/models/llama/llama-2-7b.Q2_K.gguf", 
    n_ctx = 1024,
    verbose=False,
    n_threads = 64)  

#tender_vector_store = Chroma.from_documents(documents, llamaEmbeddings) 