from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from src.service.chatbot.loaders.file_loader_factory import FileLoaderFactory


def test_get_loader_for_pdf():
    # Test if the loader for PDF files is returned correctly
    file_path = "./resources/sample.pdf"
    loader = FileLoaderFactory.get_loader(file_path)
    assert isinstance(loader, PyPDFLoader)
    assert loader.file_path == file_path


def test_get_loader_for_other_file_types():
    # Test if the loader for other file types is returned correctly
    file_path = "./resources/sample.txt"
    loader = FileLoaderFactory.get_loader(file_path)
    assert isinstance(loader, UnstructuredFileLoader)
    assert loader.file_path == file_path
