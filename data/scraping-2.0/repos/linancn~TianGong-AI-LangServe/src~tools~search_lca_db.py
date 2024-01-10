import os
import re
from typing import Optional, Type

from dotenv import load_dotenv
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel
from xata.client import XataClient

load_dotenv()


xata_api_key = os.getenv("XATA_API_KEY")
xata_db_url = os.getenv("XATA_LCA_DB_URL")
xata_branch = os.getenv("XATA_LCA_DB_BRANCH")

xata = XataClient(api_key=xata_api_key, db_url=xata_db_url)


class SearchLCADB(BaseTool):
    name = "search_lca_tool"
    description = "Use original query to search in the Life Cycle Assessment Database."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        results = xata.data().search_branch(
            branch_name=xata_branch, payload={"query": query}
        )

        docs = results["records"]

        return docs

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        results = xata.data().search_branch(
            branch_name=xata_branch, payload={"query": query}
        )

        docs = results["records"]

        return docs
