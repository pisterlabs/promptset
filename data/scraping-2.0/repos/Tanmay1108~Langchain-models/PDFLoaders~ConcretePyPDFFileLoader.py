from langchain.document_loaders import PyPDFLoader
from IDocumentLoader import IDocumentLoader

class ConcretePyPDFFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = PyPDFLoader(file, extract_images=True)
        
    def return_text(self, page_length):
        import pdb;pdb.set_trace()
        docs = self.loader.load()
        return docs[0].page_content[:page_length]