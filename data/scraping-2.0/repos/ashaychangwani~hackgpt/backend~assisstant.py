from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

class Assisstant:
    def __init__(self):
        x = 4

    def help(self, location, query):
        loader = TextLoader(location)
        index = VectorstoreIndexCreator().from_loaders([loader])
        result = index.query(query)
        return result
