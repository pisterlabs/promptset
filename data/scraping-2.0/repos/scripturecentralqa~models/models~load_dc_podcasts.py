"""Load podcasts."""

from typing import Any

from bs4 import BeautifulSoup
from bs4 import Tag
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def extract_title(soup: BeautifulSoup) -> Any:
    """Extract the title from the page."""
    # get the first section
    section = soup.find("section")
    if not isinstance(section, Tag):
        return None
    # get the second div with class elementor-col-50
    divs = section.find_all("div", class_="elementor-col-50")
    if len(divs) < 2:
        return None
    # get the third h2 from this div
    h2s = divs[1].find_all("h2")
    if len(h2s) < 3:
        return None
    # return the third h2
    return h2s[2]


def extract_content(soup: BeautifulSoup) -> Any:
    """Extract the HTML content from the page."""
    # Find all sections
    sections = soup.find_all("section")

    # check that there are at least 5 sections
    if len(sections) < 5:
        return None

    # return the sixth section
    return sections[4]


def load_dc_podcasts(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load dc podcasts from a url and html."""
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    content = extract_content(soup)

    content = clean(to_markdown(str(content), base_url=url)) if content else ""

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    return Document(page_content=content, metadata=metadata)
