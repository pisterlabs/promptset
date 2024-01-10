import openai


class GPTClient:
    def __init__(self,
                 openai_api_type: str,
                 openai_api_key: str,
                 openai_endpoint: str,
                 openai_api_version: str):
        openai.api_type = openai_api_type
        openai.api_key = openai_api_key
        openai.api_base = openai_endpoint
        openai.api_version = openai_api_version
        self._instance = openai

    def get_instance(self):
        return self._instance
