from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool
from langchain.tools.gmail.create_draft import GmailCreateDraft
from langchain.tools.gmail.get_message import GmailGetMessage
from langchain.tools.gmail.get_thread import GmailGetThread
from langchain.tools.gmail.search import GmailSearch
from langchain.tools.gmail.send_message import GmailSendMessage
from langchain.tools.gmail.utils import build_resource_service

from googleapiclient.discovery import Resource


SCOPES = ["https://mail.google.com/"]


class CustomGmailToolkit(BaseToolkit):
    """Toolkit for interacting with Gmail."""

    api_resource: Resource = Field(default_factory=build_resource_service)

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            GmailCreateDraft(api_resource=self.api_resource),
            GmailSendMessage(api_resource=self.api_resource),
            GmailSearch(api_resource=self.api_resource),
            GmailGetMessage(api_resource=self.api_resource),
            GmailGetThread(api_resource=self.api_resource),
        ]

# Update forward references after the class definition
GmailCreateDraft.update_forward_refs(Resource=Resource)
GmailSendMessage.update_forward_refs(Resource=Resource)
#GmailSearch.update_forward_refs(Resource=Resource)
GmailGetMessage.update_forward_refs(Resource=Resource)
GmailGetThread.update_forward_refs(Resource=Resource)


