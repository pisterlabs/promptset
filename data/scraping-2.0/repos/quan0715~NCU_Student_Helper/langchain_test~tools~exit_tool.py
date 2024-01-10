from langchain.tools import BaseTool
from typing import Any


class ExitTool(BaseTool):
    name = "ExitTool"
    description = "Useful when user want to exit or break this conversation in any."
    return_direct = True

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "EXIT"
