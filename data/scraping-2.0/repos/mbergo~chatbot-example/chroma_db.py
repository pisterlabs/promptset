import langchain

class ChromaDB:
    def __init__(self, path):
        self.db = langchain.ChromaDB(path)

    def index_url(self, url):
        self.db.index_url(url)
