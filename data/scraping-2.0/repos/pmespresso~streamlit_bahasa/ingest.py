"""This is the logic for ingesting Notion data into LangChain."""
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import pickle
from typing import List
from dotenv import load_dotenv

load_dotenv()


paths = ['./pdfs/Buku1.pdf', './pdfs/Buku2.pdf', './pdfs/Buku3.pdf']

documents: List[Document] = []

for p in tqdm(paths, desc="Loading and Splitting PDFs"):
    loader = PyPDFLoader(p)
    doc = loader.load_and_split()
    documents = [*documents, *doc]

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_documents(documents, OpenAIEmbeddings())
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)