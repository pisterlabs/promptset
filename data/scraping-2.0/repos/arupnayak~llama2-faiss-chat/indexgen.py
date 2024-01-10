from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

loader = PyPDFLoader("meditations.pdf")

documents = loader.load_and_split ()

text_splitter = RecursiveCharacterTextSplitter (chunk_size=600, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
faiss_index = FAISS.from_documents(texts, embeddings)
faiss_index_name = 'faiss-meditations-index'

faiss_index.save_local(faiss_index_name)