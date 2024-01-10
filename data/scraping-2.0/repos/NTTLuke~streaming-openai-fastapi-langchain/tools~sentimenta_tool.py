from langchain.tools import BaseTool


# FOR TESTING PURPOSES ONLY
class SentimentTool(BaseTool):
    """
    Tool for analyzing the sentiment of a text using the HuggingFace inference endpoint.
    """

    name = "Sentiment Tool Analysis"
    description = """
    """

    def _run(self, query: str) -> str:
        return "GOOOOOD"
