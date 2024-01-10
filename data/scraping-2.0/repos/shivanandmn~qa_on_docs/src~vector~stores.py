from langchain.vectorstores import Chroma


class LoadVector:
    def __init__(self, documents, embedding, persist_directory) -> None:
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory,
        )

    def search_relevent_docs(self, text: str, k: int, fetch_k: int):
        return self.vectordb.max_marginal_relevance_search(text, k=k, fetch_k=fetch_k)

    def __repr__(self) -> str:
        collections = self.vectordb._collection.count()
        return f"Total Collection :{collections}"
