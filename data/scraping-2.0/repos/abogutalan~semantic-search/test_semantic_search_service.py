import os
import unittest
import elasticsearch
from unittest.mock import patch
import openai

import pandas as pd
from semantic_search_service import SemanticSearchService


# Define a test class that inherits from unittest.TestCase
class TestSemanticSearchService(unittest.TestCase):

    # Define a setUp method that runs before each test
    def setUp(self):
        # Set up OpenAI API key and Elasticsearch client
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.es_client = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])
        # Create an instance of the semantic search service class
        self.semantic_search_service = SemanticSearchService(self.openai_api_key, self.es_client)
        # Define a simple sample data set of sentences
        self.df = pd.DataFrame({
            "Title": ["Chunk2Doc", "ChunkEmbeddings", "ChunkTokenizer", "TextMatcher"],
            "Text": [
                "Converts a CHUNK type column back into DOCUMENT. Useful when trying to re-tokenize or do further analysis on a CHUNK result.",
                "This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate chunk embeddings from either Chunker, NGramGenerator, or NerConverter outputs.For extended examples of usage, see the Examples and the ChunkEmbeddingsTestSpec.",
                "Tokenizes and flattens extracted NER chunks.The ChunkTokenizer will split the extracted NER CHUNK type Annotations and will create TOKEN type Annotations. The result is then flattened, resulting in a single array.For extended examples of usage, see the ChunkTokenizerTestSpec.",
                "Annotator to match exact phrases (by token) provided in a file against a Document. A text file of predefined phrases must be provided with setEntities. For extended examples of usage, see the Examples and the TextMatcherTestSpec."
            ],
            "Link": ["https://sparknlp.org/docs/en/annotators#chunk2doc", "https://sparknlp.org/docs/en/annotators#chunkembeddings", "https://sparknlp.org/docs/en/annotators#chunktokenizer", "https://sparknlp.org/docs/en/annotators#textmatcher"]

        }).reset_index()
        # Define an index name
        self.index_name = "sample"

    # Define a tearDown method that runs after each test
    def tearDown(self):
        # Delete the index if it exists
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)

    # Define a test method for the get_embedding method
    def test_get_embedding(self):
        # Define a sample text and its expected embedding
        text = "Hello world"
        text_embedding = openai.Embedding.create(input = [text], model="text-embedding-ada-002")['data'][0]['embedding']
        expected_embedding = text_embedding # A list of 1536 floats
        # Use patch to mock the OpenAI API call
        with patch("openai.Embedding.create") as mock_create:
            # Set the mock return value to match the expected embedding
            mock_create.return_value = {
                "data": [
                    {
                        "embedding": expected_embedding
                    }
                ]
            }
            # Call the get_embedding method with the sample text
            actual_embedding = self.semantic_search_service.get_embedding(text)
            # Assert that the actual embedding matches the expected embedding
            self.assertEqual(actual_embedding, expected_embedding)

    # Define a test method for the index_data method
    def test_index_data(self):
        # Call the index_data method with the sample data frame and index name
        self.semantic_search_service.index_data(self.df, self.index_name)
        # Assert that the index exists
        self.assertTrue(self.es_client.indices.exists(index=self.index_name))
        # Assert that the number of documents in the index matches the number of rows in the data frame
        self.assertEqual(self.es_client.count(index=self.index_name)["count"], len(self.df))
    
    # Define a test method for the semantic_search method
    def test_semantic_search(self):
        # Index the sample data frame before searching
        self.semantic_search_service.index_data(self.df, self.index_name)
        # Define a sample query and its expected results (a list of (score, title, text) tuples)
        query = "How can I match texts?"
        expected_results = [
            (0.88447416, "TextMatcher", "Annotator to match exact phrases (by token) provided in a file against a Document. A text file of predefined phrases must be provided with setEntities. For extended examples of usage, see the Examples and the TextMatcherTestSpec.", "https://sparknlp.org/docs/en/annotators#textmatcher"),
            (0.8577605, "ChunkEmbeddings", "This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate chunk embeddings from either Chunker, NGramGenerator, or NerConverter outputs.For extended examples of usage, see the Examples and the ChunkEmbeddingsTestSpec.", "https://sparknlp.org/docs/en/annotators#chunkembeddings"),
            (0.8391215, "ChunkTokenizer", "Tokenizes and flattens extracted NER chunks.The ChunkTokenizer will split the extracted NER CHUNK type Annotations and will create TOKEN type Annotations. The result is then flattened, resulting in a single array.For extended examples of usage, see the ChunkTokenizerTestSpec.", "https://sparknlp.org/docs/en/annotators#chunktokenizer"),
        ]
        query_embedding = openai.Embedding.create(input = [query], model="text-embedding-ada-002")['data'][0]['embedding']

        # Use patch to mock the OpenAI API call
        with patch("openai.Embedding.create") as mock_create:
            # Set the mock return value to match the expected query embedding
            mock_create.return_value = {
                "data": [
                    {
                        "embedding": query_embedding
                    }
                ]
            }
            # Call the semantic_search method with the sample query and index name
            actual_results = self.semantic_search_service.semantic_search(query, self.index_name)

            # Assert that the actual results match the expected results
            self.assertEqual(round(actual_results[0][0], 2), round(expected_results[0][0], 2))

# Run the unittest
if __name__ == "__main__":
    unittest.main()
