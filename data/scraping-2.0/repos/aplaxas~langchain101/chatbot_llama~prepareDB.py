import os
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# pip install sentence_transformers

HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

print(HUGGINGFACEHUB_API_TOKEN)

documents = []
directory_path = './data/calltext/'
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        pdf_path = directory_path+filename
        loader = TextLoader(pdf_path, encoding='utf-8')
        documents.extend(loader.load())

print(len(documents))

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

txtSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documentChunks = txtSplitter.split_documents(documents)

print(len(documentChunks))

print(documentChunks[0])

persist_directory = './db/calltext/'
vectordb = Chroma.from_documents(
    documents=documentChunks,
    embedding=embeddings,
    persist_directory=persist_directory)
print("Saved")

vectordb.persist()
vectordb = None
print("persist")
