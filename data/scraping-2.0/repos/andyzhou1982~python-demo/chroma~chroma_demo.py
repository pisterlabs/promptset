import os
import chromadb
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

def raw_get_collection(name:str)->chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="./chroma/db")
    collection = chroma_client.get_or_create_collection(name=name)
    return collection


def raw_add_from_texts(collection:chromadb.Collection, texts:list[str], metadatas:list, ids:list):
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

#chroma原始调用
def raw_exec():
    collection = raw_get_collection(name="test1")
    documents=["This is a document about engineer", "This is a document about steak","This is a document about foods"]
    metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
    ids=["id1", "id2", "id3"]
    raw_add_from_texts(collection, documents, metadatas, ids)
    results = collection.query(
        query_texts=["Which food is the best?"],
        n_results=3
    )
    print(f"results={results}")


def read_files_in_directory(path:str):
    docs = []
    for root, dirs, files in os.walk(path):   # 不是一次取出所有文件，而是一层一层取出
        for file in files:
            file_path = os.path.join(root, file)
            #print(f"file_path:{file_path}")
            file_type = file.split('.')[-1]
            if file_type == 'pdf':
                docs.extend(PyMuPDFLoader(file_path).load())
            elif file_type == 'md':
                docs.extend(UnstructuredMarkdownLoader(file_path).load())
            elif file_type == 'txt':
                docs.extend(UnstructuredFileLoader(file_path).load())
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def add_from_filepath(filepath:str, persist_directory:str):
    documents = read_files_in_directory(filepath)
    
    vectordb = Chroma.from_documents(
        documents=documents[:],
        embedding=embedding,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    vectordb.persist()

def langchain_exec():
    add_from_filepath("resources/books", "./chroma/db1")
    vectordb = Chroma(
        persist_directory="./chroma/db1",
        embedding_function=embedding
    )
    print(f"向量库中存储的数量：{vectordb._collection.count()}")


def langchain_query():
    question="指令微调"
    vectordb = Chroma(
        persist_directory="./chroma/db1",
        embedding_function=embedding
    )
    sim_docs = vectordb.similarity_search(question,k=2)
    print(f"检索到的内容：{sim_docs}")

if __name__ == "__main__":
    #raw_exec()
    #langchain_exec()
    langchain_query()