from langchain.schema import BaseOutputParser


class StringOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text.strip()
