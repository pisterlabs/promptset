# coding=utf-8

from langchain.document_loaders.base import BaseLoader
from langchain.indexes import VectorstoreIndexCreator

class IndexBuilder(object):
    def __init__(self) -> None:
        pass

    def build_index(self, loader: BaseLoader):
        index = VectorstoreIndexCreator().from_loaders([loader])
        return index
