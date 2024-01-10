from langchain.tools.base import BaseTool


class IsInHeavenTool(BaseTool):
    name = "is_in_heaven"
    description = "useful for answering questions about whether someone is in heaven"

    def _run(self, query: str, run_manager = None) -> str:
        return f"Yes, {input} he is in heaven. (Make some witty comment about him/her)"
    
    async def _arun(self, query: str, run_manager = None) -> str:
        raise NotImplementedError("custom_search does not support async")