from langchain.llms.base import LLM
import requests
import json


class Baichuan(LLM):
    @property
    def _llm_type(self) -> str:
        return "Baichuan"

    def _call(self, messages: list) -> str:
        data = {"messages": messages}
        response = requests.post(self.url, json=data)
        response = json.loads(response.content)
        response = response["response"]
        return response
