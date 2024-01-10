from abc import ABC, abstractmethod
from typing import List, Any

from langchain_core.documents import Document


class AbstractSourceListDocLoader(ABC):
    @staticmethod
    @abstractmethod
    def load_docs_from_source_list(source_list: List[Any]) -> List[Document]:
        pass
