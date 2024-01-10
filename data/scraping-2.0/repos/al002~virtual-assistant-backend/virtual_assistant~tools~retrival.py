from langchain.tools import BaseTool
from langchain.schema import BaseRetriever

class RetrivalTool(BaseTool):
    retriver: BaseRetriever

    name =  "knowledgebase retrival tool"

    description = (
        "Retrival for knowledgebase"
        "Useful when you need retrieve information and question context from local knowledgebase."
        "Input should be a query."
    )

    def _run(self, query: str):
        results =  self.retriver.get_relevant_documents(query)
        if len(results) == 1:
            return results[0].page_content

        if len(results) > 1:
            return results[0].page_content + results[1].page_content

        return ""

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("RetrivalRun does not support async") 