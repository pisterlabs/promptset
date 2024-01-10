# coding=utf-8

from .base_loader import BaseFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.base import BaseLoader
from tools.file_helper import FileHelper
from tools.logging_helper import LoggerHelper

logger = LoggerHelper().get_logger()

class PDFLoader(BaseFileLoader):
    def __init__(self, pdf_file_path):
        super().__init__(pdf_file_path)

    def build_loader(self) -> BaseLoader:
        '''
        Load one pdf file from specified folder.
        '''
        logger.debug(f'pdf file name: {self.folder_path}')
        return PyPDFLoader(f'{self.folder_path}')