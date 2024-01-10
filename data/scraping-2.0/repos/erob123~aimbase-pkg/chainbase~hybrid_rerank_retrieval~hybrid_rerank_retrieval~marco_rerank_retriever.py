from typing import Any, List
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import Callbacks
from sentence_transformers import CrossEncoder

class MarcoRerankRetriever(BaseRetriever):
    """Retriever class for Marco Rerank."""

    base_retriever: BaseRetriever
    """The initial VectorStoreRetriever used for document retrieval."""
    rerank_model_name_or_path: str | None = None
    """The name or path of the model used for reranking.  Must be a model that can be loaded by the CrossEncoder class from sentence_transformers."""
    cross_model: CrossEncoder | None = None
    """The CrossEncoder model used for reranking. Loads automatically from rerank_model_name_or_path."""
    max_relevant_documents: int | None = None
    """The maximum number of documents to return from the reranker. If None, all documents are returned."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """

        # Step 1: Use the base retriever to get initial document retrieval
        initial_docs = self.base_retriever.get_relevant_documents(query, callbacks=callbacks, **kwargs)

        # Step 2: Rerank the documents using Marco rerank method
        reranked_docs = self._marco_rerank(query, initial_docs)

        # Step 3: If max_relevant_documents is set, return only the top max_relevant_documents
        if self.max_relevant_documents is not None:
            reranked_docs = reranked_docs[: self.max_relevant_documents]

        return reranked_docs

    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        
        # Step 1: Use the base retriever to get initial document retrieval
        initial_docs = await self.base_retriever.aget_relevant_documents(query, callbacks=callbacks, **kwargs)

        # Step 2: Rerank the documents using Marco rerank method
        reranked_docs = self._marco_rerank(query, initial_docs)

        # Step 3: If max_relevant_documents is set, return only the top max_relevant_documents
        if self.max_relevant_documents is not None:
            reranked_docs = reranked_docs[: self.max_relevant_documents]

        return reranked_docs

    def _marco_rerank(self, query: str, initial_docs: List[Document]) -> List[Document]:
        """
        Implement the Marco rerank method to reorder the initial results.

        Parameters:
            query (str): The input query for retrieval.
            initial_docs (List[Dict[str, Union[str, float]]]): The initial retrieved documents.

        Returns:
            List[Document]: The reranked list of documents with their scores added to metadata,
            sorted in decreasing order.
        """

        # Step 1: check if the model is already loaded
        if self.cross_model is None:
            model_to_load = self.rerank_model_name_or_path or 'cross-encoder/ms-marco-TinyBERT-L-6'
            self.cross_model = CrossEncoder(model_to_load, max_length=512)

        # Step 2: Score the initial documents
        model_inputs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_model.predict(model_inputs)

        # Step 3: Sort the scores in decreasing order
        for doc, score in zip(initial_docs, scores):
            doc.metadata['rerank_score'] = score

        reranked_docs = sorted(initial_docs, key=lambda doc: doc.metadata['rerank_score'], reverse=True)
        return reranked_docs
