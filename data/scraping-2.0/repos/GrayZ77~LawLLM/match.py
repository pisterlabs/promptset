from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os

embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

def load_documents(directory = "/root/DISC-LawLLM/法律文书"):
    print("loading documents……")
    raw_documents = DirectoryLoader(directory).load()
    text_splitter = CharacterTextSplitter(chunk_size = 128, chunk_overlap = 0)
    docs = text_splitter.split_documents(raw_documents)
    return docs

def store_vector(docs, embeddings, persist_dirctory="VectorDataBase"):
    print("storing vectors……")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persist_dirctory)
    return db


def quest(query, num = 3):
    embedding_vector = embeddings.embed_query(query)
    if not os.path.exists("VectorDataBase"):
        documents = load_documents()
        db = store_vector(documents, embeddings)
    else:
        db = FAISS.load_local("VectorDataBase", embeddings)
    message = ''
    docs = db.max_marginal_relevance_search_by_vector(embedding_vector, k=num)
    for i, doc in enumerate(docs):
        message = message + f"{i+1}. {doc.page_content}  \n\n"
    
    return message

if __name__ == "__main__":
    print(quest(query = "什么样的行为可以被称为垄断行为", num=5))


