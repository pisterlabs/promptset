from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from pathlib import Path

class DocumentLoader:
    def load_file(self, file_path):
        documents = []

        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if file_path is not None:
            file_name = file_path.name
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            elif file_name.endswith('.docx') or file_name.endswith('.doc'):
                loader = Docx2txtLoader(str(file_path))
                documents.extend(loader.load())
            elif file_name.endswith('.txt'):
                loader = TextLoader(str(file_path))
                documents.extend(loader.load())

        return documents
