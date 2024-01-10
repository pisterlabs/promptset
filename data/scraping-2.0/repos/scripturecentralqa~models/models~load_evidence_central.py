"""Load evidence central."""

import re
from typing import Any

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def extract_title(soup: BeautifulSoup) -> Any:
    """Extract the title from the page."""
    # get the title
    title = soup.find("h1", class_="article__title")
    return title


def extract_content(soup: BeautifulSoup) -> Any:
    """Extract the HTML content from the page."""
    # Find all sections
    content = soup.find_all("div", class_="col-lg-8")
    return content


def clean_data(markdown_content: str) -> Any:
    """Remove unwanted text from markdown content."""
    # Search for the position of "abstract" (case insensitive)
    abstract_match = re.search(r"^## ABSTRACT\b", markdown_content, re.IGNORECASE | re.MULTILINE)

    if abstract_match:
        # Extract content after the "abstract" text
        abstract_position = abstract_match.start()
        markdown_content = markdown_content[abstract_position:]

    # Remove lines with the specified pattern
    markdown_content = re.sub(r"!\[\]\(/api/attachments/\d+/download\)", "", markdown_content)

    # Search for the position of "abstract" (case insensitive)
    further_match = re.search(r"^##### FURTHER READING\b", markdown_content, re.IGNORECASE | re.MULTILINE)

    if further_match:
        # Extract content before the "further reading" text
        further_position = further_match.start()
        markdown_content = markdown_content[:further_position]

    return markdown_content


def load_evidence_central(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load evidence central from a url and html."""
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    content = extract_content(soup)

    content = clean(to_markdown(str(content), base_url=url)) if content else ""

    content = clean_data(content)

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    # print(metadata)
    return Document(page_content=content, metadata=metadata)
