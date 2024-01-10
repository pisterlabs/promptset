import unittest
from langchain.embeddings import OpenAIEmbeddings
import sys
sys.path.append('.')
from src.Database import VectorDatabase

class VectorDatabaseTests(unittest.TestCase):

    def __init__(self, methodName):
        super().__init__(methodName)
        self.vdb = VectorDatabase()

    def test_epub_loader(self):
        print(self.vdb.add_epub("bookdata/Dune{Frank_Herbert}.epub"))

if __name__ == '__main__':
    unittest.main()
