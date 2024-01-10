"""Load Pearl of Great Price."""

from bs4 import BeautifulSoup
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def load_pogp(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load Pearl of Great Price from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    title = soup.find("h1", class_="entry-title")
    body = soup.find("div", class_="entry-content")

    content = clean(to_markdown(str(body), base_url=url)) if body else ""
    content = content.split("\n\n#### Further Reading\n\n")[0]
    metadata = {
        "url": url,
        "title": clean(title) if title else "",
        # "author": clean(author) if author else "",
    }
    return Document(page_content=content, metadata=metadata)
