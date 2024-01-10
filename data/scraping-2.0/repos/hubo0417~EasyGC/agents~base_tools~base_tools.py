from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


# 重写抽象方法，"call_func"方法执行 tool
class functional_Tool(BaseTool):
    name: str = ""
    description: str = ""

    def _call_func(self, query, **kwargs):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(self,
             query: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._call_func(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")
