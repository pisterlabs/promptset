import unittest
from unittest.mock import patch
from chatbot.exceptions.InternalServiceException import InternalServiceException
from chatbot.accessor.DatabaseAccessor import DatabaseAccessor
from chatbot.accessor.OpenAIAccessor import OpenAIAccessor
from chatbot.service.BotService import BotService


class TestBotService(unittest.TestCase):

    def setUp(self):
        self.bot_service = BotService()

    @patch.object(DatabaseAccessor, 'get_context_from_db')
    @patch.object(OpenAIAccessor, 'get_response')
    def test_prompt_and_respond_success(self, mock_get_response, mock_get_context_from_db):
        # Mocking the context from the database and response from OpenAI
        mock_get_context_from_db.return_value = "context information"
        mock_get_response.return_value = "Answer to the question."

        question = "What is the color of the sky?"
        response = self.bot_service.prompt_and_respond(question)

        self.assertEqual(response, "Answer to the question.")

    @patch.object(DatabaseAccessor, 'get_context_from_db', side_effect=InternalServiceException)
    @patch.object(OpenAIAccessor, 'get_response')
    def test_prompt_and_respond_with_database_exception(self, mock_get_response, mock_get_context_from_db):
        # Mocking an exception while retrieving context from the database
        mock_get_response.return_value = "Answer with error context."
        mock_get_context_from_db.side_effect = InternalServiceException("Database Error")

        question = "What is the color of the sky?"
        response = self.bot_service.prompt_and_respond(question)

        # Verify that the OpenAIAccessor was called with the error context
        mock_get_response.assert_called_with("Error: Unable to reach database", question)

        # Verify the response
        self.assertEqual(response, "Answer with error context.")
