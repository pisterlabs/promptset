from langchain.tools import BaseTool
from pydantic import root_validator
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from typing import Optional, Dict

class GoogleSerperTool(BaseTool):
    """Tool that queries the Serper.dev Google search API."""

    name = "google_serper"
    description = (
        "Useful for when you need to search the internet for recommendations on patients profile "
        "which you don't know about. Input should be a search query."
    )
    serper_api_key: str
    
    @root_validator()
    def initiate_api(cls, values: Dict) -> Dict:
        values["api_wrapper"] = GoogleSerperAPIWrapper(serper_api_key=values["serper_api_key"])
        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.api_wrapper.run(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return NotImplementedError