import re

from langchain.schema import BaseOutputParser


class MarkdownOutputParser(BaseOutputParser):
    """Parse an output using xml format."""

    tag: str = "markdown"

    def get_format_instructions(self) -> str:
        return f"""The output should be a markdown file enclosed in tags <{self.tag}> and </{self.tag}>."""

    def parse(self, text: str) -> str:
        pattern = re.compile(
            rf"<{self.tag}>(.*)</{self.tag}>", re.MULTILINE | re.DOTALL
        )
        text = re.search(pattern, text).group(1)

        return text

    @property
    def _type(self) -> str:
        return "markdown"
