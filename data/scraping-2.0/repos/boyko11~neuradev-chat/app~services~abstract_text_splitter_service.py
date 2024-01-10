from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class AbstractTextSplitterService(ABC):

    @abstractmethod
    def split_docs(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError("Must implement split_docs() method")