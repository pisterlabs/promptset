from src.utils.vectara_manager import VectaraManager
from langchain.document_loaders import UnstructuredHTMLLoader
import glob

class DocLoader:
    def __init__(self):
        self.vectara_manager = VectaraManager()

    def load_docs(self):
        '''
        Loads all docs in the docs folder
        :return:
        '''
        client = self.vectara_manager.get_vectara_client()
        docs = glob.glob("docs/*.html")
        lanchain_docs = []
        for doc in docs:
            with open(doc, "r") as f:
                loader = UnstructuredHTMLLoader(doc)
                data = loader.load()
                lanchain_docs.extend(data)

        client.add_documents(lanchain_docs)




DocLoader().load_docs()