from typing import List, Optional, Dict, Any

from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.schema.vectorstore import VectorStoreRetriever, VectorStore


class PagesRetriever(VectorStoreRetriever):

    vectorstore: VectorStore
    search_type: str
    search_kwargs: dict
    page_map: Dict

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = super()._get_relevant_documents(query, run_manager=run_manager)
        pages = [self.page_map[doc.metadata["page_num"]] for doc in docs]
        return pages
