"""
This tool allows agents to interact with the SendGrid Api
and operate on a SendGrid instance.

To use this tool, you must first set as environment variables:
    SLACK_BOT_TOKEN
"""

from typing import Optional
from pydantic import Field

from langchain.tools.base import BaseTool
from send_grid_api import SendGridApiWrapper


class SendGridAction(BaseTool):
    api_wrapper: SendGridApiWrapper = Field(default_factory=SendGridApiWrapper)
    mode: str
    name = ""
    description = ""

    def _run(self, instructions: Optional[str]) -> str:
        """Use the SendGrid API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)

    async def _arun(self, _: str) -> str:
        """Use the SendGrid API to run an operation."""
        raise NotImplementedError("SendGridAction does not support async")
