import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from lawgpt.config import settings

class vecDBService:
    def __init__(self, embedding_model: str = settings.EMBEDDING_MODEL, docs_path: str = settings.DOCS_PATH):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.docs_path = docs_path
        self.vector_store_path = settings.VECDB_PATH
        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)

    def add_document(self, document_path):
        loader = UnstructuredFileLoader(document_path, mode="elements")
        doc = loader.load()
        self.vector_store.add_documents(doc)
        self.vector_store.save_local(self.vector_store_path)

    def load_vector_store(self, path):
        if path is None:
            return None
        else:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store
        
    def get_knowledge(self, query: str, top_k: int=4):
            embeddings = self.embeddings.embed_query(query)
            docs = self.vector_store.similarity_search_by_vector(embeddings, top_k)
            return [doc.page_content for doc in docs]

def create_knowledge(src_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    docs = []
    for doc in os.listdir(src_path):
        if doc.endswith('.txt'):
            f = open(f'{src_path}/{doc}', 'r', encoding='utf-8')

            for line in f.readlines():

                docs.append(Document(page_content=''.join(
                    line.strip()), metadata={"source": f'doc_id_{doc}'}))

    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(save_path)

    

if __name__ == "__main__":
    create_knowledge('./doc', './cache/legal_article')