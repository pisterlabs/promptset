import os
import unittest
from unittest.mock import patch

import openai.types
import requests

from nimbusagent.utils import helper

os.environ["OPENAI_API_KEY"] = "some key"


class TestHelperFunctions(unittest.TestCase):

    @patch("openai.resources.Moderations.create")
    def test_is_query_safe(self, mock_moderation_create):
        categories_dict = {
            'sexual': False,
            'sexual/minors': False,
            'violence': False,
            'violence/graphic': False,
            'hate': False,
            'hate/threatening': False,
            'harassment': False,
            'harassment/threatening': False,
            'self-harm': False,
            'self-harm/instructions': False,
            'self-harm/intent': False
        }

        categories_scores = {
            'harassment': 0.0,
            'harassment/threatening': 0.0,
            'hate': 0.0,
            'hate/threatening': 0.0,
            'self-harm': 0.0,
            'self-harm/instructions': 0.0,
            'self-harm/intent': 0.0,
            'sexual': 0.0,
            'sexual/minors': 0.0,
            'violence': 0.0,
            'violence/graphic': 0.0
        }

        mock_moderation_create.return_value = openai.types.ModerationCreateResponse(
            id="mod-123",
            model="content-filter-alpha-c4",
            results=[
                openai.types.Moderation(
                    categories=openai.types.moderation.Categories(**categories_dict),
                    category_scores=openai.types.moderation.CategoryScores(**categories_scores),
                    flagged=False
                )
            ]
        )

        self.assertTrue(helper.is_query_safe("Is it going to rain today?"))

        categories_dict['violence'] = True
        categories_scores['violence'] = 0.9
        mock_moderation_create.return_value = openai.types.ModerationCreateResponse(
            id="mod-123",
            model="content-filter-alpha-c4",
            results=[
                openai.types.Moderation(
                    categories=openai.types.moderation.Categories(**categories_dict),
                    category_scores=openai.types.moderation.CategoryScores(**categories_scores),
                    flagged=True
                )
            ]
        )
        self.assertFalse(helper.is_query_safe("Some unsafe query"))

    @patch("openai.resources.Moderations.create")
    def test_is_query_safe_network_failure(self, mock_moderation_create):
        # Simulate a network failure using requests.exceptions.ConnectionError
        mock_moderation_create.side_effect = requests.exceptions.ConnectionError()

        # Call the is_query_safe function and expect it to handle the network failure gracefully
        result = helper.is_query_safe("test query")

        self.assertFalse(result)  # if openai is unavailable then the query is unsafe

    @patch("openai.resources.Embeddings.create")
    def test_get_embedding(self, mock_embedding_create):
        # First part of the test
        embedding = openai.types.Embedding(
            embedding=[0.1, 0.2],
            index=0,
            object="embedding"
        )

        mock_embedding_create.return_value = openai.types.CreateEmbeddingResponse(
            id="emb-123",
            model="text-embedding-ada-002",
            object="list",
            data=[embedding],
            usage=openai.types.create_embedding_response.Usage(
                prompt_tokens=0,
                total_tokens=0
            )
        )

        # {"data": [{"embedding": [0.1, 0.2]}]}
        self.assertEqual(helper.get_embedding("some text"), [0.1, 0.2])

        # Reset the mock
        mock_embedding_create.reset_mock()

        # Second part of the test
        mock_embedding_create.side_effect = Exception("Some error")
        self.assertIsNone(helper.get_embedding("some text"))

    def test_cosine_similarity(self):
        self.assertAlmostEqual(helper.cosine_similarity([1, 0], [0, 1]), 0)
        self.assertAlmostEqual(helper.cosine_similarity([1, 0], [1, 0]), 1)

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.2, 0.1]
        function_embeddings = [
            {'name': 'func1', 'embedding': [0.2, 0.1]},
            {'name': 'func2', 'embedding': [0.1, 0.3]}
        ]
        result = helper.find_similar_embedding_list("some query", function_embeddings)
        self.assertEqual(result[0]['name'], 'func1')

    def test_combine_lists_unique(self):
        self.assertEqual(helper.combine_lists_unique([1, 2], [2, 3]), [1, 2, 3])
        self.assertEqual(helper.combine_lists_unique([1, 2], {3, 4}), [1, 2, 3, 4])

    def test_combine_lists_unique_with_tuples(self):
        list1 = (1, 2)  # Using a tuple instead of a list
        set2 = {3, 4}  # Set can remain as it is

        # Call the combine_lists_unique function with a tuple and a set
        # Depending on your function's implementation, you might expect a specific result or an exception
        result = helper.combine_lists_unique(list1, set2)

        # Assert the expected behavior
        # If your function can handle tuples, check the resulting list
        # If not, you might use `self.assertRaises(TypeError, helper.combine_lists_unique, list1, set2)`
        self.assertEqual(result, [1, 2, 3, 4])  # Adjust this assertion based on your expected behavior

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_with_none_embeddings(self, mock_get_embedding):
        # noinspection PyTypeChecker
        result = helper.find_similar_embedding_list("some query", None)
        self.assertIsNone(result)

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_with_empty_embeddings(self, mock_get_embedding):
        result = helper.find_similar_embedding_list("some query", [])
        self.assertIsNone(result)

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_with_empty_query(self, mock_get_embedding):
        function_embeddings = [{'name': 'func1', 'embedding': [0.2, 0.1]}]
        result = helper.find_similar_embedding_list("", function_embeddings)
        self.assertIsNone(result)

    @patch("nimbusagent.utils.helper.get_embedding")
    def test_find_similar_embedding_list_get_embedding_returns_none(self, mock_get_embedding):
        mock_get_embedding.return_value = None
        function_embeddings = [{'name': 'func1', 'embedding': [0.2, 0.1]}]
        result = helper.find_similar_embedding_list("some query", function_embeddings)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
