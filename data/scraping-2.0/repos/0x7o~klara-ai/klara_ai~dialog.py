import translators.server as ts
from .endpoint import Endpoint
from .config import Config
import openai
import json
import os

import logging

logger = logging.getLogger(__name__)


class OpenAI:
    def __init__(self, config: Config):
        self.config = config
        self.endpoint = Endpoint(config)
        openai.api_key = self.config.get_config("openai_api_key")

    def get_history(self):
        history_file = self.config.get_config("history_file")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                return json.load(f)
        else:
            self.create_history()
            return self.get_history()

    def create_history(self):
        history_file = self.config.get_config("history_file")
        with open(history_file, "w") as f:
            json.dump([], f)

    def write_history(self, ai, human):
        history = self.get_history()
        history.append({"ai": ai, "human": human})
        history_file = self.config.get_config("history_file")
        with open(history_file, "w") as f:
            json.dump(history, f)

    def get_prompt(self, human):
        base_prompt = self.config.get_config("base_prompt")
        bot_name = self.config.get_config("bot_name")
        history = self.get_history()[10:]
        # history_split = self.config.get_config("history_split")
        prompt = base_prompt
        for i in range(len(history)):
            prompt += f"Human: {history[i]['human']}\n{bot_name}: {history[i]['ai']}\n"
        prompt += f"Human: {human}\n{bot_name}:"
        return prompt

    def get_response(self, human):
        logger.info("Sending request to openai")
        bot_name = self.config.get_config("bot_name")
        human = ts.google(human, from_language="ru", to_language="en")
        prompt = self.get_prompt(human)
        response = openai.Completion.create(
            engine="text-ada-001",
            prompt=ts.google(prompt, from_language="ru", to_language="en"),
            temperature=0.9,
            max_tokens=150,
            top_p=1,
            stop=["\n", " Human:", f" {bot_name}:"],
        )
        text = response["choices"][0]["text"]
        self.write_history(text, human)
        logger.info(f"Response: {text}")
        return ts.google(text, from_language="en", to_language="ru")


if __name__ == "__main__":
    config = Config("config.json")
    dialog = OpenAI(config.get_config("openai_api_key"), config)
    print(dialog.get_response("Hello"))
