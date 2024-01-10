from langchain.document_loaders import UnstructuredFileLoader
from IDocumentLoader import IDocumentLoader

class ConcreteUnstructuredFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = UnstructuredFileLoader(file)
        
    def return_text(self, page_length):
        docs = self.loader.load()
        return docs[0].page_content[:page_length]