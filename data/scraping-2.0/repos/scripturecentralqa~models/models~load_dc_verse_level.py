"""Load D&C Verse-Level Commentary."""

import re
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
    if not first_section:
        return None

    # Find the first <h2> element within the <section>
    first_h2 = first_section.find("h2")

    return first_h2


def get_content(soup: BeautifulSoup) -> Any:
    """Gets page content."""
    # Find all <section> elements in the HTML
    sections = soup.find_all("section")

    # Check if there are at least three <section> elements
    if len(sections) < 3:
        return None

    # Return the first div with class elementor-widget inside the third section
    div = sections[2].find("div", class_="elementor-widget")
    return div


def convert_verses_to_headings(content: str) -> str:
    """Convert Verse N or Verses X-Y to level 2 markdown headings."""
    content = re.sub(r"(?:^|\n) *Verse (\d+) *\n", r"\n## Verse \1\n", content)
    content = re.sub(r"(?:^|\n) *Verses (\d+)-(\d+) *\n", r"\n## Verses \1-\2\n", content)
    return content.strip()


def load_dc_verse_level(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load knowhys from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = get_title(soup)
    body = get_content(soup)
    content = clean(to_markdown(str(body), base_url=url)) if body else ""
    content = convert_verses_to_headings(content)
    content = re.sub(r"\n\s*\[\d+\]\(#t\d+\)\.[^\n]*", "", content)
    clean_content = content.replace("\n(*Doctrine & Covenants Minute*)\n\n**Casey Paul Griffiths** (LDS Scholar)\n", "")

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    return Document(page_content=clean_content, metadata=metadata)
