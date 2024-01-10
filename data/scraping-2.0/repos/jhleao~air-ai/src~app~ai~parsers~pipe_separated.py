from typing import List

from langchain.output_parsers import ListOutputParser


class PipeSeparatedListOutputParser(ListOutputParser):
    """
    LangChain only has a CommaSeparatedListOutputParser.
    This is the same thing but with pipes instead.
    For parsing entries that would have commas within them.
    """

    def get_format_instructions(self, eg: str = "foo|bar|baz") -> str:
        return f"Your response should be a list of pipe separated values, eg: `{eg}`"

    def parse(self, text: str) -> List[str]:
        parsed = text.strip().split("|")
        if len(parsed) == 1 and parsed[0] == "":
            return []
        return parsed
