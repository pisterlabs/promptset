import json

from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        return json.loads(text)
