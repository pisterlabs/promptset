
from typing import Optional 

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.tools import BaseTool 

class MusicInformationRetrievalTool(BaseTool):
    name = "MusicInformationRetrieval"
    description = "useful for when you need to analyze audio or music"

    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return "this is a test"

    async def _arun(
            self, 
            query: str, 
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("MusicInformationRetrievalTool does not support async")
