"""Load knowhys."""

import re
from typing import Any

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def clean_text(markdown_content: str) -> Any:
    """Remove unwanted texts from dc people content."""
    # init return text
    text_data = markdown_content

    # Search for the position of a footnote
    text_match = re.search(r"^\[\d+\]\(#t\d+\)\.", markdown_content, re.IGNORECASE | re.MULTILINE)

    if text_match:
        # Check content after the "text" text
        text_position = text_match.start()

        # Check if there is text after the "text" section
        if text_position > 0:
            text_data = markdown_content[:text_position]

    return text_data


def load_dc_people(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load dc people from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = soup.find("h1", class_="elementor-heading-title")

    body = soup.find("div", class_="elementor-element-7c4c46d2")
    content = clean(to_markdown(str(body), base_url=url)) if body else ""

    content = clean_text(content)

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    return Document(page_content=content, metadata=metadata)
