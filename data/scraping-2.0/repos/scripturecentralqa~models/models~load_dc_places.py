"""Load D&C Places."""

import re

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def remove_year_headers(text: str) -> str:
    """Define the regular expression pattern."""
    pattern = r"## \d{4}(-\d{4})?(\s+-{3,})?"

    # Use re.sub to replace matches with an empty string
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def places_clean(text: str) -> str:
    """Make key points a level 3 heading."""
    text = clean(text)
    text = text.replace("## Key Points", "### Key Points")
    return text


def load_dc_places(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load dc places from a url and html."""
    body = []
    soup = BeautifulSoup(html, bs_parser)
    title = clean(soup.find("div", class_="elementor-text-editor")).replace("/", "", 1)
    sections = soup.find_all("section", class_="has_eae_slider")
    for section in sections:
        if (
            not section.find("div", class_="elementor-gallery-item__content")
            and not section.find("a", class_="e-gallery-item")
            and not section.find("div", class_="elementor-background-overlay")
            and not any("Significant Events At a Glance" in tag.text for tag in section.find_all("h2"))
            and not any("D&C Sections Received Here" in tag.text for tag in section.find_all("h2"))
            and not any("Take a 360°" in tag.text for tag in section.find_all("h2"))
            and not any("Photo Credit:" in tag.text for tag in section.find_all("h2"))
            and not any("(Tap photo to enlarge and see more information)" in tag.text for tag in section.find_all("h2"))
            and not any("Directions to" in tag.text for tag in section.find_all("h2"))
            and not section.find("a", href=True)
        ):
            text = places_clean(to_markdown(str(section), base_url=url)) if section else ""
            text = remove_year_headers(text)
            # print('text:',text)
            if text == "## ":
                continue
            body.append(text)
    content = "\n\n--------\n\n".join(body)
    metadata = {
        "url": url,
        "title": clean(title) if title else "",
        # "author": clean(author) if author else "",
    }
    return Document(page_content=content, metadata=metadata)
