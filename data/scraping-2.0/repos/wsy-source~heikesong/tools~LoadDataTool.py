from typing import Any

from langchain.tools import BaseTool

from util.TempStore import TempStore


class LoadDataTool(BaseTool):
    name = "LoadDataTool"
    description = """useful tool to user all input load data to memory
                    parameter: content(user all input)
                    """

    def _run(self, content) -> Any:
        TempStore.content = content
        return "Load data to memory successfully"

    async def _arun(self, content) -> Any:
        TempStore.content = content
        return "Load data to memory successfully"
