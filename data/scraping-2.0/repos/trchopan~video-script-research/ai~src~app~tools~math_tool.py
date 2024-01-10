import sys
from io import StringIO
from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chat_models import ChatOpenAI
from langchain.tools.base import BaseTool
from pydantic import Field
from wikipedia import re


class MathTool(BaseTool):
    """Useful for when you need to answer questions about math.
This tool is only for math questions and nothing else.

Formulate the input as python code.
"""

    name = "MathTool"
    description = """Useful for when you need to answer questions about math.
This tool is only for math questions and nothing else.

Formulate the input as python code.
"""
    chat: ChatOpenAI = Field(exclude=True)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the MathTool tool."""

        _expression = f"import datetime\nimport math\nfrom math import *\n\n{query}"
        try:
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            exec(_expression)
            sys.stdout = old_stdout
            output = redirected_output.getvalue()

        except Exception as e:
            raise ValueError(
                f'LLMMathChain._evaluate("{query}") raised error: {e}.'
                " Please try again with a valid numerical expression"
            )

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool asynchronously."""
        raise NotImplementedError("WikipediaQueryRun does not support async")
