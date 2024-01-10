"""Load Historical Contexts."""

from typing import Any

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def get_title(soup: BeautifulSoup) -> Any:
    """Gets page title."""
    # Find the first <section> element
    first_section = soup.find("section")

    # Check if a <section> element was found
    if first_section:
        # Find the first <h2> element within the <section>
        first_h2 = first_section.find("h2")

        # Check if a <h2> element was found within the <section>
        if first_h2:
            return first_h2  # Return the first <h2> element
        else:
            return None  # No <h2> element found within the <section>
    else:
        return None  # No <section> element found in the HTML


def get_content(soup: BeautifulSoup) -> Any:
    """Gets page content."""
    # Find all <section> elements in the HTML
    sections = soup.find_all("section")

    # Check if there are at least two <section> elements
    if len(sections) < 2:
        return None

    # Get the divs with class elemenetor-col-50
    divs = sections[1].find_all("div", class_="elementor-col-50")

    # Check if there are at least two divs with class elemenetor-col-50
    if len(divs) < 2:
        return None

    # Return the second div
    return divs[1]


def load_dc_historical_context(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load Historical Context from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = get_title(soup)
    body = get_content(soup)
    content = clean(to_markdown(str(body), base_url=url)) if body else ""
    content = content.split("[1](#t1).")[0]

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    return Document(page_content=content, metadata=metadata)
