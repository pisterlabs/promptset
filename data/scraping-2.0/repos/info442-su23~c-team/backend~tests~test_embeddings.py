from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
import unittest
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.getenv("PYTHONPATH")))
from src.embeddings import *

class TestLangChain(unittest.TestCase):
    def setUp(self):
        print("Setting up the test environment...")
        self.messages = ["Hello world", "Goodbye world"]
        self.message_id_name = "label"
        self.text_name = "test_text"
        self.embedder = OpenAIEmbeddings(model='text-embedding-ada-002')
        self.vector_store = Chroma()

    def test_embed(self):
        print("Testing the embed function...")
        embeddings_dict = embed(self.messages, self.message_id_name, self.text_name, self.embedder)
        self.assertEqual(len(embeddings_dict), len(self.messages))
        for i, message_id in enumerate(embeddings_dict):
            self.assertEqual(embeddings_dict[message_id]["text"], self.text_name)
            self.assertEqual(embeddings_dict[message_id]["type"], self.message_id_name)
    
    def test_embed_meta_separate(self):
        print("Testing the embed_meta_separate function...")
        embeddings_dict = embed_meta_separate(self.messages, self.message_id_name, self.text_name, self.embedder)
        self.assertEqual(len(embeddings_dict["embeddings"], len(self.messages)))
        for i, _ in enumerate(self.messages):
            self.assertEqual(embeddings_dict["metadatas"]["text"], self.text_name)
            self.assertEqual(embeddings_dict["metadatas"]["type"], self.message_id_name)
    
    #Chroma doesn't support single-dict embeddings, therefore embed_meta_separate is used
    def test_store(self):
        print("Testing the store function...")
        embeddings_dict = embed_meta_separate(self.messages, self.message_id_name, self.text_name, self.embedder)
        store(embeddings_dict, self.text_name, self.vector_store)
        all_embeddings = self.vector_store.fetch_all()
        self.assertEqual(len(all_embeddings), len(self.messages))

    def test_retrieve(self):
        print("Testing the retrieve function...")
        embeddings_dict = embed_meta_separate(self.messages, self.message_id_name, self.text_name, self.embedder)
        store(embeddings_dict, self.text_name, self.vector_store)
        retrieved_texts = retrieve(self.text_name, self.message_id_name, vector_store=self.vector_store)
        self.assertEqual(len(retrieved_texts), len(self.messages))

    def test_semantic_search(self):
        print("Testing the semantic_search function...")
        embeddings_dict = embed_meta_separate(self.messages, self.message_id_name, self.text_name, self.embedder)
        store(embeddings_dict, self.text_name, self.vector_store)
        search_results = semantic_search("world", self.text_name, self.message_id_name, vector_store=self.vector_store, embedder=self.embedder)
        self.assertTrue(len(search_results) > 0)

    def tearDown(self):
        print("Cleaning up after the tests...")
        self.vector_store.delete_collection()

if __name__ == '__main__':
    unittest.main()