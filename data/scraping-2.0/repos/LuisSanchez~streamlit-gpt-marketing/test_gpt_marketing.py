import datetime
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice


class TestGPTMarketing(unittest.TestCase):
    # See https://github.com/openai/openai-python/issues/715#issuecomment-1809203346
    def create_chat_completion(
        self, response: str, role: str = "assistant"
    ) -> ChatCompletion:
        return ChatCompletion(
            id="foo",
            model="gpt-4-vision-preview",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=response,
                        role=role,
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
        )

    @patch("openai.resources.chat.Completions.create")
    def test_gpt_marketing(self, openai_create):
        PROMPT = "This is a simple test for the GPT-4 vision model."

        at = AppTest.from_file("gpt_marketing.py").run()
        assert not at.exception
        at.chat_input[0].set_value(PROMPT).run()
        assert at.info[0].value == "Please add your OpenAI API key to continue."
        at.text_input(key="chatbot_api_key").set_value("sk-...")
        at.chat_input[0].set_value(PROMPT).run()
        assert at.info[0].value == "Please add an image URL to continue."

        openai_create.return_value = self.create_chat_completion(PROMPT)
        at.text_input(key="url_key").set_value("https://...")
        at.chat_input[0].set_value(PROMPT).run()

        assert at.chat_message[1].markdown[0].value == PROMPT
        assert at.chat_message[1].avatar == "user"
        assert at.chat_message[2].avatar == "assistant"
        assert not at.exception
