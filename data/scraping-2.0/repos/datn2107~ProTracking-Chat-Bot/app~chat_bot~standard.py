from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from .base import ChatBotBase


class ChatBotStandard(ChatBotBase):
    def __init__(
        self, debug=False
    ):
        search = GoogleSearchAPIWrapper()
        self.tools = [
            Tool(
                name="Google Search",
                description="Search Google for recent results.",
                func=search.run,
            )
        ]
        super().__init__(debug=debug)
