from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

persist_directory = './db/calltext/'
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

query = "DPS# 748700319"
docs = vectordb.similarity_search(query)
print(docs)
