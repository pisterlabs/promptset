"""Utility functions for loading and saving data."""

import json
import os
import re
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import cast

from bs4 import Tag
from langchain.document_loaders.base import BaseLoader
from langchain.schema.document import Document
from markdownify import MarkdownConverter  # type: ignore
from tqdm import tqdm


def clean(text: Any) -> str:
    """Convert text to a string and clean it."""
    if text is None:
        return ""
    if isinstance(text, Tag):
        text = text.text
    if not isinstance(text, str):
        text = str(text)
    """Replace non-breaking space with normal space and remove surrounding whitespace."""
    text = text.replace(" ", " ").replace("\u200b", "").replace("\u200a", " ")
    text = re.sub(r"(\n\s*)+\n", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    return cast(str, text.strip())


def to_markdown(html: str, base_url: str) -> str:
    """Convert html to markdown."""
    return cast(
        str,
        MarkdownConverter(
            heading_style="ATX",
            strip=["script", "style"],
            base_url=base_url,
        ).convert(html),
    )


def save_docs_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    """Save documents to jsonl file."""
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + "\n")


def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    """Load documents from jsonl file."""
    array = []
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


def create_pages_from_unstructured_elements(
    elements: list[Document], title: str, start_page: int = 0, url: str = ""
) -> list[Document]:
    """Create pages from unstructured elements."""
    pages = []
    page_content = ""
    page_number = 0
    for element in elements:
        # skip some elements
        if element.metadata["category"] in ["Title", "PageBreak", "UncategorizedText"]:
            continue
        # skip pages before start_page
        if element.metadata.get("page_number", 0) < start_page:
            continue
        # get element content
        element_content = element.page_content
        if element.metadata["category"] == "ListItem":
            element_content = "* " + element_content
        if len(element_content.strip()) == 0:
            continue
        if element.metadata.get("page_number", 0) != page_number:
            if len(page_content) > 0:
                # Create a page
                title_with_page = f"{title} - {page_number}" if page_number > 0 else title
                page = Document(
                    page_content=(page_content + "\n" + element_content).strip(),
                    metadata={
                        "title": title_with_page,
                        "url": url,
                        "page_number": page_number,
                    },
                )
                pages.append(page)
                page_content = ""
            else:
                page_content = element_content
            page_number = element.metadata.get("page_number", 0)
            continue
        page_content += "\n\n" + element_content
    if len(page_content) > 0:
        # Create a page
        title_with_page = f"{title} - {page_number}" if page_number > 0 else title
        page = Document(
            page_content=page_content.strip(),
            metadata={
                "title": title_with_page,
                "url": url,
                "page_number": page_number,
            },
        )
        pages.append(page)
    return pages


class Loader(BaseLoader):
    """Create Documents for all files in a directory."""

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement lazy_load()")

    def __init__(self, load_fn: Callable[[str, str, str], Document], path: str = "", bs_parser: str = "html.parser"):
        """Initialize loader."""
        super().__init__()
        self.load_fn = load_fn
        self.path = path
        self.bs_parser = bs_parser

    def load(self, verbose: bool = False) -> list[Document]:
        """Load documents from path."""
        docs = []
        for filename in tqdm(os.listdir(self.path), disable=not verbose):
            path = os.path.join(self.path, filename)
            with open(path, encoding="utf8") as f:
                data = json.load(f)
            doc = self.load_fn(data["url"], data["html"], self.bs_parser)
            if not doc.metadata["title"] or not doc.page_content:
                if verbose:
                    print("Missing title or content - skipping", data["url"])
                continue
            docs.append(doc)
        return docs
