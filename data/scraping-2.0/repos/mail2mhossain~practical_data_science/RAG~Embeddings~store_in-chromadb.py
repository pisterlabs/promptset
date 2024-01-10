import os
import glob

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader  # for textfiles
from langchain.text_splitter import CharacterTextSplitter  # text splitter
from langchain.embeddings import HuggingFaceEmbeddings  # for using HugginFace models

# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import (
    FAISS,
)  # facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader  # load pdf


persist_directory = "../ChromaDB"
docs = []

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Text Document Loader
txt_folder_path = "../Documents/"
txt_files = glob.glob(os.path.join(txt_folder_path, f"*{'.txt'}"))
txt_docs = [text_splitter.split_documents(TextLoader(fn).load()) for fn in txt_files]
docs.extend(txt_docs)

# PDF loader
pdf_folder_path = "../Documents/"
# os.listdir(pdf_folder_path)
pdf_files = glob.glob(os.path.join(pdf_folder_path, f"*{'.pdf'}"))

pdf_docs = [
    text_splitter.split_documents(UnstructuredPDFLoader(fn).load()) for fn in pdf_files
]
docs.extend(pdf_docs)

# for fn in pdf_files:
#     loader = UnstructuredPDFLoader(fn)
#     documents = loader.load()
#     documents = text_splitter.split_documents(documents)
#     docs.extend(documents)


# Initialize PeristedChromaDB
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="corporate_db",
    persist_directory=persist_directory,
)

# Persist the Database
vectordb.persist()
vectordb = None
