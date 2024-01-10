"""Tools for searching documents for the most relevant one."""
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from pydantic import BaseModel

from .base_tool import ToolTemplate


def search_documents_tool(db: Chroma, sources: list) -> Tool:
    """Search tool using ChromaDB to search documents for the most relevant one."""

    def _run(query: str) -> Document:
        _docs = db.similarity_search(query)
        answer = _docs[0]
        return answer

    def _arun(query: str) -> None:
        raise NotImplementedError("This tool does not support async")

    return Tool.from_function(
        func=_run,
        name="Search Documents Tool",
        description=f"Useful for searching information if you don't know the answer. Contains information about things such as {','.join([s.name for s in sources])} Input should be a search query.",
    )


class SearchDocumentTool(ToolTemplate):
    """Tool for searching documents for the most relevant one."""

    name: str = "Search Documents Tool"
    user_description: str = "use this tool to sent data to a server."

    user_variables: list = []

    @property
    def args_schema(self) -> BaseModel:
        """Return the args schema for langchain."""

        class ArgsSchema(BaseModel):
            query: str

        return ArgsSchema

    def __init__(self, db: Chroma, sources: list) -> None:
        """Initialize the tool."""
        self.db = db
        self.sources = sources

    @property
    def bot_description(self) -> str:  # type: ignore
        """Return the tool description for llm."""
        return f"Useful for searching information if you don't know the answer. Contains information about things such as {','.join([s.name for s in self.sources])}"

    def run(self, *args: list, **kwargs: dict) -> str:
        """Run the tool by sending a post request to the url with the body."""
        query = kwargs.get("query", None) or args[0]
        _docs = self.db.similarity_search(query)
        answer = _docs[0].page_content
        return answer
