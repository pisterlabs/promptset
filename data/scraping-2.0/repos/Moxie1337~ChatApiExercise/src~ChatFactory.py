import openai
import json

from pathlib import Path
from src.BackoffWrapper import retry_with_exponential_backoff

class ChatFactory:
    def __init__(self, openai_conf_path, messages, model = "gpt-3.5-turbo"):

        self.openai_conf_ = json.loads(Path(openai_conf_path).read_text())
        self.api_base_ = self.openai_conf_.get("api_base")
        self.api_key_ = self.openai_conf_.get("api_key")

        self.model_ = model
        self.messages_ = messages
        self.completion = None

        self.construct_openai_conf()

    def construct_openai_conf(self):
        openai.api_base = self.api_base_
        openai.api_key = self.api_key_

    @retry_with_exponential_backoff
    def create_chat(self):
        self.completion = openai.ChatCompletion.create(
            model=self.model_,
            messages=self.messages_
        )

    def get_chat_info(self):
        return self.completion.get("choices")[0].message.content
