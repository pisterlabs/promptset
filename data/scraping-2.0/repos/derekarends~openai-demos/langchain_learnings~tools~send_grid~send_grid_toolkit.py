"""SendGrid Toolkit."""
from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from send_grid_tool import SendGridAction
from send_grid_api import SendGridApiWrapper


class SendGridToolKit(BaseToolkit):
    """SendGrid Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_send_grid_api_wrapper(cls, send_grid_api_wrapper: SendGridApiWrapper) -> "SendGridToolKit":
        actions = send_grid_api_wrapper.list()
        tools = [
            SendGridAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=send_grid_api_wrapper,
            )
            for action in actions
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
