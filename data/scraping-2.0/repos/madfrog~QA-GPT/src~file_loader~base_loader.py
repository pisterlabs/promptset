# coding=utf-8

from langchain.document_loaders.base import BaseLoader
from abc import ABC, abstractmethod

class BaseFileLoader(ABC):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.loader = None

    @abstractmethod
    def build_loader(self) -> BaseLoader:
        pass
        
