import os
from abc import ABC, abstractmethod
from typing import Literal, Any, List
import logging
import requests
from selenium import webdriver
from bs4 import BeautifulSoup, ResultSet, PageElement
from langchain.schema import Document
from langchain.document_loaders.unstructured import UnstructuredBaseLoader

from dotenv import load_dotenv
load_dotenv()


def decompose_many(elements: ResultSet[PageElement]) -> None:
    """
    Decomposes all elements in ResultSet of PageElement objects produced by BeautifulSoup query

    :param elements: The ResultSet containing one or many PageElement objects
    :return: None
    """
    for e in elements:
        e.decompose()

class UnstructuredHtmlStringLoader(UnstructuredBaseLoader):
    """
    Custom loader for HTML strings using UnstructuredBaseLoader
    """
    def __init__(self, content: str, source: str = None, mode: str = "single", **unstructured_kwargs: Any):
        self.content = content
        self.source = source
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> list:
        from unstructured.partition.html import partition_html

        return partition_html(text=self.content, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.source} if self.source else {}

class PageFilter(ABC):
    """
    Abstract PageFilter class
    PageFilters can be configured for Scraper class and will be used to filter found pages before preprocessing and returning the result
    """
    def __init__(self, sites: List[str]):
        self.sites = sites

    """
    Abstract filter method that must be implemented in PageFilter classes and defines the filtering outcome

    :param html: The HTML string to check for filter criteria
    :return: The bool signalling fulfillment of filter criteria
    """
    @abstractmethod
    def filter(self, html: str) -> bool:
        raise NotImplementedError

class MediumMemberOnlyPageFilter(PageFilter):
    """
    PageFilter for Medium that filters member-only pages
    """
    def __init__(self):
        sites = ["medium.com"]
        super().__init__(sites)
    
    def filter(self, html: str) -> bool:
        soup = BeautifulSoup(html, "html.parser")
        if soup.find("p", string="Member-only story"):
            return True

        return False
    
class HtmlPreprocessor(ABC):
    """
    Abstract HTML Preprocessor class
    HTML Preprocessors can be configured for Scraper class and will be used to preprocess HTML of pages after filtering and before returning the result
    """
    def __init__(self, sites: List[str]):
        self.sites = sites

    """
    Abstract preprocess method that must be implemented in Preprocessor classes and defines the preprocessing outcome

    :param html: The HTML string to preprocess
    :return: The preprocessed and potentially reduced HTML string
    """
    @abstractmethod
    def preprocess(self, html: str) -> str:
        raise NotImplementedError
    
class MediumFooterRemovalHtmlPreprocessor(HtmlPreprocessor):
    """
    Preprocessor for Medium that removes the footer and post recommendations
    """
    def __init__(self):
        sites = ["medium.com"]
        super().__init__(sites)

    def preprocess(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        footer = soup.find("footer")
        if footer:
            footer_next_siblings = footer.find_next_siblings()
            decompose_many([footer, *footer_next_siblings])

        return str(soup)

class Scraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.browser = os.environ.get("BROWSER", "chrome")
        self.logger.info(f"Using browser {self.browser}")
        self.browser_binary_location = os.environ.get("BROWSER_BINARY_LOCATION", None)
        self.logger.info(f"Using browser binary location {self.browser_binary_location if self.browser_binary_location else 'default'}")

        self.driver = None
        if self.browser == "firefox":
            from selenium.webdriver.firefox.options import Options
            options = Options()
            options.binary_location = self.browser_binary_location
            options.headless = True
            options.add_argument("--no-sandbox")
            self.driver = webdriver.Firefox(options=options)
        elif self.browser == "chrome":
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.binary_location = self.browser_binary_location
            options.headless = True
            options.add_argument("--no-sandbox")
            self.driver = webdriver.Chrome(options=options)

        self.session = requests.Session()
        self.duckduckgo_search_url = "https://lite.duckduckgo.com/lite/"
        self.sites = ["medium.com"]
        self.page_filters = [MediumMemberOnlyPageFilter]
        self.html_preprocessors = [MediumFooterRemovalHtmlPreprocessor]

    def filter_page(self, site: str, html: str) -> bool:
        for Filter in self.page_filters:
            filter = Filter()
            if site in filter.sites:
                if filter.filter(html):
                    return True
        
        return False
    
    def preprocess(self, site: str, input: str) -> str:
        output = ""
        for Preprocessor in self.html_preprocessors:
            preprocessor = Preprocessor()
            if site in preprocessor.sites:
                output = preprocessor.preprocess(input)

        return output

    def search_pages(self, site: str, query: str) -> list:
        query_string = f"site:{site} {query}"
        data = {"q": query_string}
        req = requests.Request(method="POST", url=self.duckduckgo_search_url, data=data)
        req = req.prepare()

        resp = self.session.send(req)

        soup = BeautifulSoup(resp.text, 'html.parser')
        results = soup.css.select("a.result-link")
        pages = [{"title": result.string, "url": result.attrs['href']} for result in results]
        self.logger.info(f"Found {len(pages)} pages for site {site}\n{os.linesep.join([page['url'] for page in pages])}")

        try:
            return pages
        except IndexError:
            pass

    def scrape(self, query: str, n_per_site: int) -> List[Document]:
        documents = []
        
        for site in self.sites:
            pages = self.search_pages(site, query)
            page_urls = [page['url'] for page in pages]

            scraped_count = 0
            for url in page_urls:
                if scraped_count >= n_per_site: break

                self.driver.get(url)
                html = self.driver.page_source

                if self.filter_page(site, html):
                    self.logger.info(f"Filtered page {url} for site {site}")
                    continue
                
                html_len_before = len(html)
                html = self.preprocess(site, html)
                self.logger.info(f"Html Preprocessing reduced page {url} for site {site} from {html_len_before} to {len(html)} characters")

                loader = UnstructuredHtmlStringLoader(content=html, source=url)
                document = loader.load()[0]
                documents.append(document)

                scraped_count = scraped_count + 1

        return documents
