import unittest
import os
import openai

from custom_nodes.SWAIN.text import get_openai_embedding


class TestGetOpenAIEmbedding(unittest.TestCase):
    def setUp(self):
        # Set up api_key
        os.environ["OPENAI_API_KEY"] = "insert_your_api_key_here"

    def test_valid_input(self):
        # Show available models
        models = openai.Model.list()
        print([model.id for model in models['data']])

        # Test valid input
        model = "text-embedding-ada-002"
        text = "Hello, world!"
        embeddings = get_openai_embedding(model, text)
        self.assertIsInstance(embeddings, list)
        self.assertIsInstance(embeddings[0][0], float)

    def test_invalid_model(self):
        # Test invalid model
        model = "text-embedding-dne"
        text = "This shouldn't work"
        try:
            embeddings = get_openai_embedding(model, text)
        except openai.error.InvalidRequestError as e:
            self.assertTrue(True)


    def test_empty_text(self):
        # Test empty text
        model = "text-embedding-ada-002"
        text = ""
        embeddings = get_openai_embedding(model, text)
        self.assertIsInstance(embeddings, list)
        self.assertIsInstance(embeddings[0][0], float)


if __name__ == "__main__":
    unittest.main()
