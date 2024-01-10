"""Load fair."""

from bs4 import BeautifulSoup
from bs4 import Tag
from langchain.schema.document import Document

from models.load_utils import clean
from models.load_utils import to_markdown


def load_fairs(url: str, html: str, bs_parser: str = "html.parser") -> Document:
    """Load fairs from a url and html."""
    soup = BeautifulSoup(html, bs_parser)
    # author = clean(soup.find("div", class_="field-nam-author")).replace("Post contributed by", "")
    # date = soup.find("div", class_="field-name-publish-date")
    # citation = soup.find(id="block-views-knowhy-citation-block")
    body = soup.find("div", id="mw-content-text")
    if not isinstance(body, Tag):
        body = soup.find("div", class_="entry-content")
    # content = soup.find(...)
    if isinstance(body, Tag):
        title = body.find("span", class_="mw-headline")
        if not isinstance(title, Tag):
            title = soup.find("h1", class_="entry-title")
    else:
        title = None
    if isinstance(title, Tag):
        title.extract()
    content = clean(to_markdown(str(body), base_url=url)) if body else ""

    metadata = {
        "url": url,
        "title": clean(title) if title else "",
        # "author": clean(author) if author else "",
        # "date": clean(date) if date else "",
        # "citation": clean(to_markdown(str(citation), base_url=url)) if citation else "",
    }
    return Document(page_content=content, metadata=metadata)
