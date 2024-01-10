from langchain.document_loaders import PDFMinerLoader
from IDocumentLoader import IDocumentLoader

class ConcretePDFMinerFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = PDFMinerLoader(file)
        
    def return_text(self, page_length):
        import pdb;pdb.set_trace()
        docs = self.loader.load()
        return docs[0].page_content[:page_length]
