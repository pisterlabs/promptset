import requests
from langchain.tools import tool
from bs4 import BeautifulSoup


@tool
def get_url(url: str) -> str:
    """Obtains information with only the text extracted from the content specified by the URL."""
    return get_url_content(url)


@tool
def get_url_head(url: str) -> str:
    """Extracts only 300 characters of content specified by URL. Use it when you just want to understand the content."""
    content = get_url_content(url)
    return content[:300]


def get_url_content(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()
