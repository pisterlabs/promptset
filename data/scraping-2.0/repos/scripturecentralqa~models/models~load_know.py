"""Load knowhys."""

import re

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


line = "* [1.](#footnoteref1"


def remove_text_below_footnote(line: str, text: str) -> str:
    """This function removes footnotes."""
    pattern = re.compile(r"\* \[1\.\]\(#footnoteref1.*\)")
    match = re.search(pattern, text)

    if match:
        # Remove everything after the matched line
        cleaned_text = text[: match.start()]
        return cleaned_text

    # If no match is found, return the original text
    return text


def load_knowhy(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load knowhys from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = soup.find("h1", class_="page-title")
    author = clean(soup.find("div", class_="field-nam-author")).replace("Post contributed by", "")
    date = soup.find("div", class_="field-name-publish-date")
    citation = soup.find(id="block-views-knowhy-citation-block")
    body = soup.find("div", class_="group-left")
    content = clean(to_markdown(str(body), base_url=url)) if body else ""
    clean_content = remove_text_below_footnote(line, content)

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
        "author": clean(author) if author else "",
        "date": clean(date) if date else "",
        "citation": clean(to_markdown(str(citation), base_url=url)) if citation else "",
    }
    return Document(page_content=clean_content, metadata=metadata)
