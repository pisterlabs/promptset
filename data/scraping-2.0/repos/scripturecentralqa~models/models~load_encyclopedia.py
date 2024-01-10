"""Load encyclopedia."""
import re

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


line = "[A](/index.php?"


def remove_stuff_words(line: str, text: str) -> str:
    """This function removes stuff words from the page content."""
    pattern = re.compile(r"\[A\]\(/index\.php\?")
    match = re.search(pattern, text)

    if match:
        # Remove everything after the matched line
        cleaned_text = text[: match.start()]
        return cleaned_text

    # If no match is found, return the original text
    return text


def load_encyclopedia(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load encyclopedia from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = soup.find("span", class_="mw-page-title-main")
    body = soup.find("div", class_="mw-parser-output")
    content = clean(to_markdown(str(body), base_url=url)) if body else ""
    clean_content = remove_stuff_words(line, content)

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
    }
    return Document(page_content=clean_content, metadata=metadata)
