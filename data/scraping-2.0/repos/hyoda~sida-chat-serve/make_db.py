from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "./data/"
DB_DIR = "./db/"

chunk_size = 300 #500  # 1000
chunk_overlap = 100 #100  # 20

def split_docs(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

documents = DirectoryLoader(DATA_DIR, glob="카페/*.txt", loader_cls=TextLoader).load()
doc = split_docs(documents)
print(len(doc))

documents = DirectoryLoader(DATA_DIR, glob="음식점/*.txt", loader_cls=TextLoader).load()
doc += split_docs(documents)
print(len(doc))

documents = DirectoryLoader(DATA_DIR, glob="달리기-런닝/*.txt", loader_cls=TextLoader).load()
doc += split_docs(documents)
print(len(doc))

documents = DirectoryLoader(DATA_DIR, glob="펜션/*.txt", loader_cls=TextLoader).load()
doc += split_docs(documents)
print(len(doc))

documents = DirectoryLoader(DATA_DIR, glob="서핑/*.txt", loader_cls=TextLoader).load()
doc += split_docs(documents)
print(len(doc))

documents = DirectoryLoader(DATA_DIR, glob="기타/*.txt", loader_cls=TextLoader).load()
doc += split_docs(documents)
print(len(doc))

documents = DirectoryLoader(DATA_DIR, glob="블로그/*.txt", loader_cls=TextLoader).load()
doc += split_docs(documents)
print(len(doc))

db = Chroma.from_documents(
    documents=doc,
    embedding=HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts"),
    persist_directory=DB_DIR,
)
db.persist()