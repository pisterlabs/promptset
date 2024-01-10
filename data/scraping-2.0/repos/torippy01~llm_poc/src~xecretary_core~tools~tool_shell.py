from langchain.tools import ShellTool


class ShellAndSummarizeTool(ShellTool):
    def _run(self, query: str) -> str:
        ## summarize
        return super()._run(query)
