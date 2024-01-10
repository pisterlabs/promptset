from langchain.document_loaders import JSONLoader
from IDocumentLoader import IDocumentLoader

class ConcreteJSONFileLoader(IDocumentLoader):
    def __init__(self, file):
        super().__init__(file)
        self.loader = JSONLoader(
            file_path=file,
            jq_schema='.messages[].content',
            text_content=False)
    
    def return_text(self, page_length):
        docs = self.loader.load()
        import pdb;pdb.set_trace()
        return docs[0].page_content[:page_length]
