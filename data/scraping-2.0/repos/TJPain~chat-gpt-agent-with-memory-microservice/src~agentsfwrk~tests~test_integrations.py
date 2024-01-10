import unittest
from unittest.mock import patch, Mock
from agentsfwrk.exceptions import APIResponseError, MaxRetriesExceededError
from agentsfwrk.integrations import OpenAIIntegrationService
from openai.error import APIError


class TestOpenAIIntegrationService(unittest.TestCase):
    def setUp(self):
        self.service = OpenAIIntegrationService(
            context="Hello, how can I assist you?", instruction="Help the user."
        )

    @patch("agentsfwrk.integrations.openai.Model.list")
    def test_get_models(self, mock_list):
        mock_list.return_value = {"models": ["model1", "model2"]}
        result = self.service.get_models()
        self.assertEqual(result, {"models": ["model1", "model2"]})

    def test_add_chat_history(self):
        self.service.add_chat_history([{"role": "user", "content": "Tell me a joke."}])
        self.assertEqual(
            self.service.messages,
            [
                {"role": "system", "content": "Hello, how can I assist you?"},
                {"role": "user", "content": "Tell me a joke."},
            ],
        )

    @patch("agentsfwrk.integrations.openai.ChatCompletion.create")
    def test_answer_to_prompt(self, mock_create):
        mock_create.return_value = Mock(
            choices=[
                Mock(message={"content": "Sure, why did the chicken cross the road?"})
            ]
        )
        result = self.service.answer_to_prompt(
            model="text-davinci-002", prompt="Tell me a joke."
        )
        self.assertEqual(
            result, {"answer": "Sure, why did the chicken cross the road?"}
        )

    @patch("agentsfwrk.integrations.openai.Completion.create")
    def test_answer_to_simple_prompt(self, mock_create):
        mock_create.return_value = Mock(
            choices=[Mock(text='{"answer": "To get to the other side."}')]
        )
        result = self.service.answer_to_simple_prompt(
            model="text-davinci-002", prompt="Why did the chicken cross the road?"
        )
        self.assertEqual(result, {"answer": "To get to the other side."})

    @patch("agentsfwrk.integrations.openai.ChatCompletion.create")
    def test_answer_to_prompt_api_error(self, mock_create):
        mock_create.side_effect = APIError("An API error occurred")
        with self.assertRaises(MaxRetriesExceededError):
            self.service.answer_to_prompt(
                model="text-davinci-002", prompt="Tell me a joke."
            )

    def test_add_chat_history_invalid_input(self):
        with self.assertRaises(ValueError):
            self.service.add_chat_history("This is not a list of dictionaries")

    @patch("agentsfwrk.integrations.openai.ChatCompletion.create")
    def test_answer_to_prompt_empty_response(self, mock_create):
        mock_create.return_value = Mock(choices=[])
        with self.assertRaises(APIResponseError):
            self.service.answer_to_prompt(
                model="text-davinci-002", prompt="Tell me a joke."
            )

    @patch("agentsfwrk.integrations.openai.ChatCompletion.create")
    def test_answer_to_prompt_retry_logic(self, mock_create):
        mock_create.side_effect = [
            APIError("An API error occurred"),
            Mock(
                choices=[
                    Mock(
                        message={"content": "Sure, why did the chicken cross the road?"}
                    )
                ]
            ),
        ]
        result = self.service.answer_to_prompt(
            model="text-davinci-002", prompt="Tell me a joke."
        )
        self.assertEqual(
            result, {"answer": "Sure, why did the chicken cross the road?"}
        )


if __name__ == "__main__":
    unittest.main()
