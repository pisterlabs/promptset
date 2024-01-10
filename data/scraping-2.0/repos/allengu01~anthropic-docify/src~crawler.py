from abc import ABC, abstractmethod
import logging
from typing import List, cast
import requests
from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter
from custom_types import Source, SourceType, Metadata

from langchain.document_loaders import UnstructuredURLLoader
from unstructured.cleaners.core import clean, clean_extra_whitespace


logger = logging.getLogger(__name__)


class Crawler(ABC):
    @abstractmethod
    def generate_row(self, url) -> Source:
        """Generates a row that contains the dataclass."""
        pass


class WebpageCrawler(Crawler):
    def __init__(self, source_type: SourceType, use_unstructured=True) -> None:
        super().__init__()
        self.source_type = source_type
        self.use_unstructured = use_unstructured

    def _get_webpage_body(self, url: str) -> Tag:
        """Uses BeautifulSoup4 to fetch a webpage's HTML body given a URL"""
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch the webpage. Status code: {response.status_code}"
            )
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        parent = soup.find("article")
        if not parent:
            raise Exception(f"No article tag found for url {url}")
        main_content = parent.find("div", class_="markdown")
        return cast(Tag, main_content)

    def _html_to_markdown(self, body: Tag) -> str:
        return MarkdownConverter().convert_soup(body)

    def generate_row(self, url: str) -> Source:
        logging.info("Starting webpage crawling")
        if self.use_unstructured:
            res = Source(
                url=url,
                content=self._get_unstructured_document(url),
                metadata=Metadata(
                    source_type=self.source_type,
                ),
            )
        else:
            res = Source(
                url=url,
                content=self._html_to_markdown(self._get_webpage_body(url)),
                metadata=Metadata(
                    source_type=self.source_type,
                ),
            )
        logger.info("Finished webpage crawling")
        return res

    def _get_unstructured_document(self, url):
        "Given an URL, return a langchain Document to futher processing"
        loader = UnstructuredURLLoader(
            urls=[url],
            mode="elements",
            post_processors=[clean, clean_extra_whitespace],
        )
        elements = loader.load()
        selected_elements = [
            e for e in elements if e.metadata["category"] == "NarrativeText"
        ]
        full_clean = " ".join([e.page_content for e in selected_elements])
        return full_clean
