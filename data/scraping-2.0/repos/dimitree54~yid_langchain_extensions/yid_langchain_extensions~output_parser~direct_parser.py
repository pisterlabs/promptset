from langchain.schema import BaseOutputParser


class DirectParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text

    def get_format_instructions(self) -> str:
        return ""
