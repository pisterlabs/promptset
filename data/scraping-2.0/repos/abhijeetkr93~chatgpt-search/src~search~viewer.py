# viewer.py
import sys
import os
import json
import openai

lib_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(lib_dir)

from choices import ModelTypes, CODE_KEYWORDS

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
except Exception as err:
    print(f"error: {err}")


class ConfigService:
    """
    create openai api payload by configuring right model based on the type of text search query
    """

    def __init__(self, model=ModelTypes.TEXT_LONG_QA.value) -> None:
        """
        :param model: ModelTypes (Q&A, ML, Code)
        """

        self.model = model
        self._cfg = {model: {}}
        path = os.path.abspath(os.path.dirname(__file__))
        with open(f"{path}/models.toml", "rb") as conf:
            self._cfg = tomllib.load(conf)

    def get_conf(self) -> dict:
        """returns model config details based"""

        return {
            "model": self._cfg[self.model].get("model"),
            "temperature": self._cfg[self.model].get("temperature", 0),
            "max_tokens": self._cfg[self.model].get("max_tokens", 100),
            "top_p": self._cfg[self.model].get("top_p", 1),
            "frequency_penalty": self._cfg[self.model].get("frequency_penalty", 0.0),
            "presence_penalty": self._cfg[self.model].get("presence_penalty", 0.0),
            "prompt": self._cfg[self.model].get("prompt", "Greeting!"),
        }

    @staticmethod
    def get_model_type(query: str) -> ModelTypes:
        """
        try to allocate similar model for query

        :param query: question asked in prompt
        :return: model types
        """

        is_code_query = any([True for key in query.split(" ") if key in CODE_KEYWORDS])
        if is_code_query:
            return ModelTypes.CODE.value
        elif "?" in query:
            return (
                ModelTypes.TEXT_QA.value
                if len(query) < 20
                else ModelTypes.TEXT_LONG_QA.value
            )
        else:
            return ModelTypes.TEXT_COMPLETION.value


class ChatGPTService:
    UNKNOWN = "Unknown"

    def __init__(self, key=None):
        self.key = key
        self.service_ok = False
        self._initialise_key()
        self.conf_service = None

    def _initialise_key(self):
        try:
            if self.key:
                openai.api_key = self.key
                self.service_ok = True
            else:
                print("please set openai key env var: CHATGPT_KEY")
        except Exception as err:
            print(f"invalid key: {err}")

    @staticmethod
    def create_prompt(question: str) -> str:
        """
        create prompt for search query in models
        """
        payload = {
            "query": f"{question}",
            "num_results": 2,
            "filters": {
                "date_range": {"start": "2022-01-01", "end": "2023-12-31"},
                "language": "en",
            },
            "sort_by": "relevance",
        }
        return json.dumps({"prompt": "search", "query": payload})

    def search_query(self, query: str) -> str:
        """
        execute search query to openai engine
        """

        try:
            conf = self.conf_service.get_conf()
            conf["prompt"] += f"{query}"
            response = openai.Completion.create(**conf)
            response = response["choices"][0]["text"]
            if response == ChatGPTService.UNKNOWN:
                self.conf_service = ConfigService(ModelTypes.TEXT_ML.value)
                conf = self.conf_service.get_conf()
                response = openai.Completion.create(**conf)
                response = response["choices"][0]["text"]
            return response
        except Exception as err:
            print(f"error: {err}")
            return "Unknown"
