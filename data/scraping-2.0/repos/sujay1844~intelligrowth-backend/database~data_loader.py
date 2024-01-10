from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

data_root = "./data"
docs = DirectoryLoader(data_root).load()

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=500,
  chunk_overlap=200,
)

esops_documents = text_splitter.transform_documents(docs)