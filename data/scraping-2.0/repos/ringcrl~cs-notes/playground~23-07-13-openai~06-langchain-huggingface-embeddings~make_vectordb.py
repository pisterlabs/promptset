from os import path
import json

from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

chunk_size = 500
chunk_overlap = 100
db_name = path.join(path.dirname(__file__), "./vectordb/docs_{chunk_size}_{chunk_overlap}".format(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap))

docs_path = path.join(path.dirname(__file__), "../00-assets/docs.json")
with open(docs_path, 'r') as f:
    # 将 JSON 文件解析为 Python 对象
    docs = json.load(f)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap)

chunked_docs = []
for doc in docs:
    chucked_doc = text_splitter.create_documents(texts=[doc["text"]], metadatas=[
        {"title": doc["title"], "source": doc["source"]}])
    chunked_docs.append(chucked_doc)


embeddings = HuggingFaceEmbeddings()

vectordb = Chroma(persist_directory=db_name,
                  embedding_function=embeddings)

for index in range(len(chunked_docs)):
    print("chunked_docs len:{docslen}, index:{index}".format(
        docslen=len(chunked_docs), index=index))
    for chunk in chunked_docs[index]:
        vectordb.add_texts(texts=[chunk.page_content],
                           metadatas=[chunk.metadata])


try:
    del vectordb
except Exception as e:
    print(e)
