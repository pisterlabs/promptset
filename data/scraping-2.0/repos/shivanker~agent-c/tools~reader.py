from typing import Type

import trafilatura
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Field

MAX_RESULT_LENGTH_CHAR = 1000 * 4 * 4  # roughly 4,000 tokens


def page_result(text: str, cursor: int, max_length: int) -> str:
    """Page through `text` and return a substring of `max_length` characters starting from `cursor`."""
    return text[cursor : cursor + max_length]


def get_url(url: str) -> str:
    """Fetch URL and return the contents as a string."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError("Could not download article.")
    return trafilatura.extract(downloaded, include_links=True, include_tables=True)


class SimpleReaderToolInput(BaseModel):
    url: str = Field(..., description="URL of the website to read")


class SimpleReaderTool(BaseTool):
    """Browser tool for getting webpage contents, with URL as the only argument."""

    name: str = "fetch_page"
    args_schema: Type[BaseModel] = SimpleReaderToolInput
    description: str = "Use this tool to fetch the contents of a webpage."

    def _run(self, url: str) -> str:
        page_contents = get_url(url)

        if len(page_contents) > MAX_RESULT_LENGTH_CHAR:
            return page_result(page_contents, 0, MAX_RESULT_LENGTH_CHAR)

        return page_contents

    async def _arun(self, url: str) -> str:
        raise NotImplementedError


class ReaderToolInput(BaseModel):
    url: str = Field(..., description="URL of the website to read")
    cursor: int = Field(
        default=0,
        description="Start reading from this character."
        "Use when the first response was truncated"
        "and you want to continue reading the page.",
    )


class ReaderTool(BaseTool):
    """Browser tool for getting webpage contents, in a paginated fashion."""

    name: str = "fetch_page"
    args_schema: Type[BaseModel] = ReaderToolInput
    description: str = "Use this tool to fetch the contents of a webpage."

    def _run(self, url: str, cursor: int = 0) -> str:
        page_contents = get_url(url)

        if len(page_contents) > MAX_RESULT_LENGTH_CHAR:
            page_contents = page_result(page_contents, cursor, MAX_RESULT_LENGTH_CHAR)
            page_contents += f"\nPAGE WAS TRUNCATED. TO CONTINUE READING, USE CURSOR={cursor+len(page_contents)}."

        return page_contents

    async def _arun(self, url: str) -> str:
        raise NotImplementedError
