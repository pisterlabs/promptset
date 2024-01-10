import os
from typing import Optional, List

from langchain.chains import MultiRetrievalQAChain
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient


def create_vs_from_docs(base_dir) -> Qdrant:
    print(f'create vectorstore from docs: base_dir = {base_dir}')
    documents = []
    loader: Optional[BaseLoader] = None
    print(f"开始读取文件夹:{os.path.abspath(base_dir)}")
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
        elif file.endswith('.txt'):
            loader = TextLoader(file_path)
        if loader is not None:
            documents.extend(loader.load())
    print("开始分割文档")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    print("开始创建矢量DB")
    vectorstore = Qdrant.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(),
        # location=":memory:",
        path="./db",
        collection_name="my_documents",
    )
    return vectorstore


def create_vs_from_exist() -> Optional[Qdrant]:
    print(f'create_vs_from_exist: ')
    client = QdrantClient(path="./db")
    collections = client.get_collections().collections
    if len(collections) == 0:
        return None
    collection_name = collections[0].name
    return Qdrant(client=client,
                  collection_name=collection_name,
                  embeddings=OpenAIEmbeddings(),
                  )


qdrant: Qdrant = create_vs_from_exist()
if not qdrant:
    qdrant = create_vs_from_docs('./OneFlower')

result: list[Document] = qdrant.similarity_search('总经理说了什么')
for d in result:
    print(d)

MultiRetrievalQAChain()