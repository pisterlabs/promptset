from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from typing import List


class DocumentRelevanceSorter(object):
    """Encapsulates the sorting of a list of Document objects by relevance."""

    def __init__(
        self,
        documents: List[Document],
        top_k: int = None,
    ) -> None:
        """
        Initialize the DocumentRelevanceSorter.

        Params:
            documents (List[Document]): The list of Document objects to sort.
            top_k (int): The number of documents to return.
        """
        self.documents = documents
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def get_sorted_by_relevance_to_query(self, query: str) -> List[Document]:
        """
        Sort the list of Document objects by relevance to a query.

        Params:
            query (str): The query to sort by.

        Returns:
            List[Document]: The sorted list of Document objects.
        """
        document_ids = [str(i) for i in range(1, len(self.documents) + 1)]
        if len(document_ids) == 0:
            return []  # no documents
        chroma_db = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            ids=document_ids,
        )
        retriever = chroma_db.as_retriever(
            search_kwargs={
                "k": len(self.documents) if self.top_k is None else self.top_k
            }
        )
        relevant_documents = retriever.get_relevant_documents(query=query)
        chroma_db._collection.delete(ids=document_ids)
        return relevant_documents
