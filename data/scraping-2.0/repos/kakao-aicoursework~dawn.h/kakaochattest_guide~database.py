from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

class DataBase:
    def __init__(self, chroma_persist_dir, chroma_collection_name):
        self.db = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=OpenAIEmbeddings(),
            collection_name=chroma_collection_name,
        )
        self.retriever = self.db.as_retriever()

    def query_db(self, query: str, use_retriever: bool = False) -> list[str]:
        if use_retriever:
            docs = self.retriever.get_relevant_documents(query)
        else:
            docs = self.db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]
        return str_docs