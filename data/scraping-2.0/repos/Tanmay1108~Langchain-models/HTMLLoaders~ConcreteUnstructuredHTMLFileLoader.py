from langchain.document_loaders import UnstructuredHTMLLoader
from IDocumentLoader import IDocumentLoader

class ConcreteUnstructuredHTMLFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = UnstructuredHTMLLoader(file)
        
    def return_text(self, page_length):
        docs = self.loader.load()
        import pdb;pdb.set_trace()
        return docs[0].page_content[:page_length]
