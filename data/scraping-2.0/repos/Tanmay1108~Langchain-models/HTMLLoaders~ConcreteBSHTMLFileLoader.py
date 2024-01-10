
from langchain.document_loaders import BSHTMLLoader
from IDocumentLoader import IDocumentLoader

class ConcreteBSHTMLFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = BSHTMLLoader(file)
        
    def return_text(self, page_length):
        docs = self.loader.load()
        import pdb;pdb.set_trace()
        return docs[0].page_content[:page_length]
