import os
from ai.agent_tools.utilities.abstract_tool import AbstractTool
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper


class InternetSearchTool(AbstractTool):
    def configure(
        self, memory=None, override_llm=None, json_args=None
    ) -> None:
        api_key = os.environ["GOOGLE_API_KEY"]
        cse_id = os.environ["GOOGLE_CSE_ID"]
        self.search = GoogleSearchAPIWrapper(
            google_api_key=api_key, google_cse_id=cse_id
        )

    def run(self, query: str) -> str:
        return self.search.run(query)
