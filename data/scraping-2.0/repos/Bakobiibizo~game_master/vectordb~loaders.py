from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    PythonLoader,
    DirectoryLoader,
    PlaywrightURLLoader,
)
from enum import Enum
from loguru import logger
from vectordb.selector import BaseSelector


class LoaderOptions(Enum):
    PYPDF_LOADER: PyPDFLoader
    UNSTRUCTURED_MARKDOWN_LOADER: UnstructuredMarkdownLoader
    TEXT_LOADER: TextLoader
    PYTHON_LOADER: PythonLoader
    DIRECTORY_LOADER: DirectoryLoader
    PLAYWRIGHT_LOADER: PlaywrightURLLoader


class LoaderSelector(BaseSelector):
    def __init__(self, default_loader=LoaderOptions.PYPDF_LOADER):
        super().__init__()
        logger.info("Initializing LoaderSelector")
        self.py_pdf_loader = LoaderOptions.PYPDF_LOADER
        self.unstructured_markdown_loader = LoaderOptions.UNSTRUCTURED_MARKDOWN_LOADER
        self.text_loader = LoaderOptions.TEXT_LOADER
        self.python_loader = LoaderOptions.PYTHON_LOADER
        self.directory_loader = LoaderOptions.DIRECTORY_LOADER
        self.playwright_url_loader = LoaderOptions.PLAYWRIGHT_LOADER
        self.loader = default_loader
        self.initialize_maps(list(LoaderOptions))

    def get_py_pdf_loader(self):
        return self.select("pypdf_loader")

    def get_unstructured_markdown_loader(self):
        return self.select("unstructured_markdown_loader")

    def get_text_loader(self):
        return self.select("text_loader")

    def get_python_loader(self):
        return self.select("python_loader")

    def get_directory_loader(self):
        return self.select("directory_loader")

    def get_playwright_url_loader(self):
        return self.select("playwright_url_loader")
