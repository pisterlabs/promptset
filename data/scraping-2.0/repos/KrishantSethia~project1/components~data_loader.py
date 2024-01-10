from langchain.document_loaders import NewsURLLoader
from typing import List
from llama_index import Document
from llama_index import download_loader
from pathlib import Path


class DataLoader:
    def data_loader_langchain(urls: List[str]):

        loader = NewsURLLoader(urls=urls)
        data = loader.load()

        llama_documents = []
        for i in range(len(data)):
            llama_document = Document()
            llama_document.text = data[i].page_content
            llama_document.metadata = data[i].metadata
            # Set other attributes as needed
            llama_documents.append(llama_document)

        return llama_documents

    def data_loader_llama_index(file_path: str):

        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=Path(file_path))
        return documents
