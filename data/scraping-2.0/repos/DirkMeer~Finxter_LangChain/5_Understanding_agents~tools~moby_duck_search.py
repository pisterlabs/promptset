from json import dumps
from langchain.tools import BaseTool
from langchain.utilities import DuckDuckGoSearchAPIWrapper


class MobyDuckSearch(BaseTool):
    name: str = "moby_duck_search"  # Pun intended.
    description: str = (
        "A tool that uses DuckDuckGo Search to search the MobyGames game website. "
        "Useful for when you need to answer questions about games. "
        "Input should be a search query. "
    )
    api_wrapper = DuckDuckGoSearchAPIWrapper()

    def _run(self, query: str) -> str:
        """Just call the DuckDuckGoSearchAPIWrapper.run method, but with the edited query."""
        targeted_query = f"site:mobygames.com {query}"
        results_with_metadata: list = self.api_wrapper.results(
            targeted_query, num_results=3
        )
        return dumps(results_with_metadata)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support asynchronous execution")


if __name__ == "__main__":
    moby_duck_tool = MobyDuckSearch()
    print(moby_duck_tool.run("lego star wars"))
