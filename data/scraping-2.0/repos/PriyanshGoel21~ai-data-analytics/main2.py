from langchain.document_loaders import TextLoader

loader = TextLoader("rag.txt")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
