import asyncio
from typing import Optional

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from pydantic import Field


class ProxyDuckDuckGoSearchRun(DuckDuckGoSearchRun):
    name = "Proxy DuckDuckGo Search"
    description = (
        "A wrapper around DuckDuckGo Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: DuckDuckGoSearchAPIWrapper = Field(
        default_factory=DuckDuckGoSearchAPIWrapper
    )

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        loop = asyncio.get_event_loop()
        # Run sync_function in a separate thread
        sync_task = loop.run_in_executor(None, self.api_wrapper.run, query)

        # Wait for both tasks to complete
        result = await asyncio.gather(sync_task)
        if len(result) > 0:
            return result[0]
        return ""
