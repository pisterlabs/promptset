import openai
import pytest

from nynoflow.chats.chat_objects import ChatMessage
from nynoflow.tokenizers import OpenAITokenizer
from tests.conftest import ConfigTests


class TestOpenaiTokenizer:
    """Test the openai tokenizer."""

    models = [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4",
    ]

    messages = list[ChatMessage](
        [
            ChatMessage(
                provider_id="chatgpt",
                role="system",
                content="You are an assistant",
            ),
            ChatMessage(
                provider_id="chatgpt",
                role="user",
                content="How are you?",
            ),
            ChatMessage(
                provider_id="chatgpt",
                role="assistant",
                content="Very good thank you. How can I help you?",
            ),
        ]
    )
    chatgpt_messages = [
        {
            "role": msg.role,
            "content": msg.content,
        }
        for msg in messages
    ]

    def test_token_count(self, config: ConfigTests) -> None:
        """Test tokenizer for models without function support."""
        for model in self.models:
            response = openai.ChatCompletion.create(
                api_key=config["OPENAI_API_KEY"],
                model=model,
                messages=self.chatgpt_messages,
                temperature=0,
                max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
            )
            usage_tokens = response["usage"]["prompt_tokens"]

            tokenizer = OpenAITokenizer(model)
            calculated_tokens = tokenizer.token_count(self.messages)
            assert usage_tokens == calculated_tokens

    def test_invalid_model(self, config: ConfigTests) -> None:
        """Make sure the tokenizer raises an exception for invalid model names."""
        with pytest.warns():
            OpenAITokenizer("invalid-model-name")
