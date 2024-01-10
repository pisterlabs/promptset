from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

class ContextManager:
    def __init__(self, store_path):
        self.store_path = store_path
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_db = None

    def load_db(self):
        self.chroma_db = Chroma(persist_directory=self.store_path, embedding_function=self.embedding_function)
        print(">>> context manager - db loaded: ", self.chroma_db)

    def get_matches(self, query):
        matches = self.chroma_db.similarity_search(query, k=10)
        matches_str = "\n".join([f"{m.page_content}" for m in matches])
        return matches_str