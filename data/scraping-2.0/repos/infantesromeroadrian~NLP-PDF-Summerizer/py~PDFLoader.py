
from langchain.document_loaders import PyPDFLoader

class PDFLoader:
    def __init__(self, filenames):
        self.filenames = filenames

    def load(self):
        documents = []
        for filename in self.filenames:
            loader = PyPDFLoader(filename)
            documents.extend(loader.load())
        return documents
