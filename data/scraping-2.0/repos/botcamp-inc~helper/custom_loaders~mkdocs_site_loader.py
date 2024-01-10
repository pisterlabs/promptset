import asyncio
from typing import Optional, List
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class MkDocsSiteLoader(BaseLoader):
    """Load files from a MkDocs site.
    Description
    -------
    This loader uses httpx to request the information from the site and then process markdown files recursively.

    Examples
    --------
    from langchain.document_loaders import MkDocsSiteLoader
    loader = MkDocsSiteLoader(
        site_url="https://docs.hummingbot.org",
        sections_filter=["/blog/", "/release-notes/", "/botcamp/", "/academy/",
                         "/academy-content/", "/exchanges/", "/chain/"],
        continue_on_failure=True,
    )
    docs = loader.load()

    """

    def __init__(
            self,
            site_url: str,
            sections_filter: Optional[List[str]] = None,
            metadata_filter: Optional[List[str]] = None,
    ):
        """Initialize with file path."""
        self.site_url = site_url
        self.sections_filter = sections_filter
        self.processed_data = []  # Global list to store results
        self.metadata_filter = metadata_filter or []

    async def aget_html(self, url, client):
        response = await client.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')

    def get_html(self, url, client):
        response = client.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')

    def get_titles_related_urls_and_paths(self, soup):
        links = [link['href'] for link in soup.find_all('a', href=True)]
        titles = '\n'.join(set(link for link in links if link.startswith("#") and "codelineno" not in link))
        related_paths = self.filter_paths(links)
        related_urls = set(link for link in links if link.startswith("http"))
        return titles, related_urls, related_paths

    @staticmethod
    def filter_paths(strings):
        path_like = []
        for string in strings:
            # Check if the string is a URL
            if string.startswith("http://") or string.startswith("https://"):
                continue
            # Check for strings that look like paths
            if "/" in string or string.endswith('/'):
                path_like.append(string)
        return path_like

    @staticmethod
    def get_text(soup):
        return soup.get_text(separator="/n", strip=True)

    @staticmethod
    def normalize_path(base_url, path):
        """
        Normalize a path to create an absolute URL.

        :param base_url: The base URL of the site.
        :param path: The path to normalize.
        :return: A normalized, absolute URL.
        """
        # If the path is already an absolute URL, return as is
        if urlparse(path).netloc:
            return path
        # Join the base URL and the path
        normalized_path = urljoin(base_url, path)
        return normalized_path

    async def arecursive_load_site(self, base_url, path, lock, client):
        processed_urls = [data.get("url") for data in self.processed_data]
        url = self.normalize_path(base_url, path)
        async with lock:
            if url in processed_urls:
                return None
        try:
            soup = await self.aget_html(url, client)
            titles, related_urls, related_paths = self.get_titles_related_urls_and_paths(soup)
            html_processed = self.get_text(soup)
            data = {
                "metadata": {
                    'url': url,
                    'related_paths': related_paths,
                    'titles': titles,
                    'related_urls': related_urls,
                },
                'page_content': html_processed,
            }
            self.processed_data.append(data)
            # Process subpaths
            for subpath in related_paths:
                next_url = self.normalize_path(url, subpath)
                # Re-evaluate this variable to catch updates from different tasks
                processed_urls = [data.get("metadata").get("url") for data in self.processed_data]
                if next_url not in processed_urls and (self.sections_filter and not any(section in next_url
                                                                                        for section in
                                                                                        self.sections_filter)):
                    await self.arecursive_load_site(url, subpath, lock, client)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            data = {
                "metadata": {
                    'url': url,
                    'related_paths': [],
                    'titles': "",
                    'related_urls': [],
                },
                'page_content': str(e),
            }
            self.processed_data.append(data)

    def recursive_load_site(self, base_url, path, client):
        processed_urls = [data.get("url") for data in self.processed_data]
        url = self.normalize_path(base_url, path)
        if url in processed_urls:
            return None
        try:
            soup = self.get_html(url, client)
            titles, related_urls, related_paths = self.get_titles_related_urls_and_paths(soup)
            html_processed = self.get_text(soup)
            data = {
                "metadata": {
                    'url': url,
                    'related_paths': related_paths,
                    'titles': titles,
                    'related_urls': related_urls,
                },
                'page_content': html_processed,
            }
            self.processed_data.append(data)
            # Process subpaths
            for subpath in related_paths:
                next_url = self.normalize_path(url, subpath)
                processed_urls = [data.get("metadata").get("url") for data in self.processed_data]
                if next_url not in processed_urls and (self.sections_filter and not any(section in next_url
                                                                                        for section in
                                                                                        self.sections_filter)):
                    self.recursive_load_site(url, subpath, client)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            data = {
                "metadata": {
                    'url': url,
                    'related_paths': [],
                    'titles': "",
                    'related_urls': [],
                },
                'page_content': str(e),
            }
            self.processed_data.append(data)

    async def aload(self):
        lock = asyncio.Lock()
        async with httpx.AsyncClient() as client:
            await self.arecursive_load_site(base_url=self.site_url, path="/", lock=lock, client=client)
        return [Document(page_content=doc["page_content"],
                         metadata={key: value for key, value in doc["metadata"].items() if
                                   key not in self.metadata_filter}
                         ) for doc in self.processed_data]

    def load(self):
        with httpx.Client() as client:
            self.recursive_load_site(base_url=self.site_url, path="/", client=client)
        return [Document(page_content=doc["page_content"],
                         metadata={key: value for key, value in doc["metadata"].items() if
                                   key not in self.metadata_filter}
                         ) for doc in self.processed_data]
