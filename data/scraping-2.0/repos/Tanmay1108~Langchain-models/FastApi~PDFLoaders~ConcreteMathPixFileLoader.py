from langchain.document_loaders import MathpixPDFLoader
from PDFLoaders.IDocumentLoader import IDocumentLoader

class ConcreteMathpixPDFFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = MathpixPDFLoader(file)
        
    def return_text(self, page_length):
        docs = self.loader.load()
        return docs[0].page_content[:page_length]
