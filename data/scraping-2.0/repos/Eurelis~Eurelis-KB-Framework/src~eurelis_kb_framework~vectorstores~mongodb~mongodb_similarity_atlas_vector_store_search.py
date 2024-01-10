from typing import Optional, Dict, List, Tuple, Callable

from langchain.schema import Document
from langchain.vectorstores import MongoDBAtlasVectorSearch


class MongoDBSimilarityAtlasVectorStoreSearch(MongoDBAtlasVectorSearch):
    """
    Class to enable similarity search with score on mongodb
    """

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        """Override to fix an issue
        TypeError: MongoDBAtlasVectorSearch.similarity_search_with_score() takes 2 positional arguments but 3 were given

        Args:
            query:
            k:
            pre_filter:
            post_filter_pipeline:

        Returns:

        """
        return super().similarity_search_with_score(
            query, k=k, pre_filter=pre_filter, post_filter_pipeline=post_filter_pipeline
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        return lambda x: x
