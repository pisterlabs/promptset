import re

from langchain.prompts import load_prompt
from langchain.schema import BaseOutputParser
from util import get_prompt_file_path


class CodeOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        match = re.search(
            r"^\`\`\`\w*$(.*)^\`\`\`$",
            text.strip(),
            re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )
        return match.group(1) if match else ""

    def get_format_instructions(self) -> str:
        return load_prompt(
            get_prompt_file_path("code_output_parser_format.yaml")
        ).format()

    @property
    def _type(self) -> str:
        return "code"
