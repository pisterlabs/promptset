import re

from langchain.schema import BaseOutputParser, OutputParserException

code_regex = re.compile(r"\n?```[a-zA-z0-9]*\n(.+?)\n```", re.DOTALL)


class CodeOutputParser(BaseOutputParser[str]):
    def get_format_instructions(self) -> str:
        """Format instructions"""
        return (
            "Write only a single markdown code block containing the requested code, "
            "without any additional formatting, commentary or explanation."
        )

    def parse(self, response: str) -> str:
        """Parse response based on markdown code regex

        Args:
            response (str): response from GPT

        Raises:
            OutputParserException: The codeblock couldn't be found.

        Returns:
            str: Code found in response codeblock
        """
        try:
            match = code_regex.match(response)
            if match is None:
                raise ValueError("Response does not match regex.")

            return match.group(1)

        except ValueError as e:
            raise OutputParserException(f"Could not parse response: {response}") from e
