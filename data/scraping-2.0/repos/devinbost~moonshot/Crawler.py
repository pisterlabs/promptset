import asyncio
import time
from typing import Tuple, List
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import requests
from newspaper import Article
from langchain.docstore.document import Document
import aiohttp
from streamlit.delta_generator import DeltaGenerator
from DataAccess import DataAccess
import logging
from datetime import datetime

from Config import config
from pydantic import BaseModel
from typing import List

from pydantic_models.PageContent import PageContent


class Crawler:
    def __init__(self, data_access: DataAccess):
        """
        Initialize the Crawler with a DataAccess object.
        Parameters:
            data_access (DataAccess): The data access object to interact with the database.
        """
        self.ui_update_in_progress = False
        self.data_access = data_access
        self.urls = None
        self.counter = 0
        self.counter_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(
            20
        )  # limit to 10 concurrent tasks, adjust as needed
        self.sitemap_gen_semaphore = asyncio.Semaphore(20)

    def get_url_count(self) -> int:
        """
        Returns the count of URLs that the crawler has processed or will process.
        Returns:
            int: The count of URLs.
        """
        return len(self.urls)

    def get_sitemap_urls(self, sitemap_url: str, onlyEnglish: bool) -> List[str]:
        """
        Retrieves URLs from a given sitemap.
        Parameters:
            sitemap_url (str): The URL of the sitemap to crawl.
            onlyEnglish (bool): Flag to indicate whether to retrieve only English URLs.
        Returns:
            List[str]: A list of URLs from the sitemap.
        """
        # Need to fix issue where it doesn't detect if onlyEnglish has been changed and will use the wrong cache file.
        urls = None  # self.load_urls_from_file() # Always do clean pull until we have smarter caching.
        if urls is None:
            now = datetime.now()
            # Convert the current time to a timestamp in milliseconds
            timestamp_ms = int(now.timestamp() * 1000)

            # Convert the timestamp to a string
            timestamp_str = str(timestamp_ms)
            r = requests.get(sitemap_url)
            soup = BeautifulSoup(r.text, "xml")
            if onlyEnglish:
                urls = [
                    loc.text for loc in soup.find_all("loc") if "locale" not in loc.text
                ]
            else:
                urls = [loc.text for loc in soup.find_all("loc")]
            with open(f"logs/urls.txt-{timestamp_str}", "w") as file:
                for url in urls:
                    file.write("%s\n" % url)
        return urls

    async def extract_page_content(
        self, url: str, session: ClientSession
    ) -> PageContent | None:
        """
        Asynchronously extracts page content from a given URL using an HTTP session.
        Parameters:
            url (str): The URL from which to extract content.
            session (ClientSession): The HTTP client session for making requests.
        Returns:
            PageContent | None: The extracted page content or None if extraction fails.
        """
        logging.info(f"Extracting content from URL: {url}")
        try:
            async with session.get(url) as response:
                html_content = await response.text()

            # Use newspaper's Article for parsing
            article = Article(url)
            article.set_html(html_content)
            article.parse()
            content = article.text
            title = article.title
            logging.info(f"Successfully extracted content from URL: {url}")
            article.nlp()
            keywords = article.keywords
            summary = article.summary
            page_content = PageContent(
                url=url,
                content=content,
                title=title,
                keywords=keywords,
                summary=summary,
            )
            return page_content
        except Exception as e:
            print(f"Timeout or some other error extracting URL {url}: {e}")
            logging.error(f"Timeout or some other error extracting URL {url}: {e}")
            return None

    async def async_chunk_page(self, url: str, session: ClientSession) -> PageContent:
        """
        Asynchronously chunks a page content from a given URL.
        Parameters:
            url (str): The URL from which to chunk content.
            session (ClientSession): The HTTP client session for making requests.
        Returns:
            PageContent: The chunked page content.
        """
        page_content = await self.extract_page_content(url, session)
        if page_content is not None:
            chunks = self.data_access.splitter.split_text(page_content.content)
            page_content.chunks = chunks
            return page_content

    async def handle_url(
        self,
        url: str,
        progress_bar: DeltaGenerator,
        session: ClientSession,
        table_name: str,
    ):
        """
        Asynchronously handles processing of a single URL.
        Parameters:
            url (str): The URL to process.
            progress_bar (DeltaGenerator): Streamlit UI element for progress indication.
            session (ClientSession): The HTTP client session for making requests.
            table_name (str): The name of the table where data is to be stored.
        """
        async with self.semaphore:  # this will wait if there are already too many tasks running:
            page_content = await self.async_chunk_page(url, session)
            if page_content is not None:
                if len(page_content.chunks) == 1 and len(page_content.chunks[0]) < 500:
                    print(
                        f"Skipping page {url} with only 1 chunk under 500 characters."
                    )
                    logging.info(
                        f"Skipping page {url} with only 1 chunk under 500 characters."
                    )
                else:
                    (
                        path_segments,
                        subdomain,
                    ) = page_content.extract_url_hierarchy_and_subdomain()
                    page_docs = [
                        Document(
                            page_content=chunk,
                            metadata={
                                "url": url,
                                "title": page_content.title,
                                "nlp_keywords": page_content.keywords_as_csv(),
                                "nlp_summary": page_content.summary,
                                "subdomain": subdomain if subdomain is not None else "",
                                **{
                                    f"path_segment_{i}": segment
                                    for i, segment in enumerate(path_segments, start=1)
                                },
                            },
                        )
                        for chunk in page_content.chunks
                    ]
                    # async transform_documents is not available yet
                    split_docs = self.data_access.splitter.transform_documents(
                        page_docs
                    )
                    vector_store = self.data_access.getVectorStore(
                        table_name=table_name
                    )
                    # vector_store.aadd_documents(split_docs) isn't yet implemented. We will work around it.
                    # await vector_store.aadd_documents(split_docs)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, vector_store.add_documents, split_docs
                    )
                    logging.info(
                        f"Written to database, URL: {url}, Title: {page_content.title}"
                    )

        async with self.counter_lock:
            self.counter += 1
            # Only call the UI update if no update is currently in progress to avoid blocking threads with UI updates since it's okay if some get skipped:
            if not self.ui_update_in_progress:
                self.ui_update_in_progress = True
            await self.update_UI(self.counter, progress_bar)

    async def update_UI(self, counter: int, progress_bar: DeltaGenerator):
        """
        Asynchronously updates the user interface with the progress of crawling.
        Parameters:
            counter (int): The current count of processed URLs.
            progress_bar (DeltaGenerator): Streamlit UI element for progress indication.
        """
        total_url_count = self.get_url_count()
        percentage_completion = ((counter + 1) / total_url_count) * 100
        print(f"Completed {counter} out of {total_url_count} in total")
        print(f"Processing... Completion: {percentage_completion:.2f}%")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        progress_bar.progress(
            int(percentage_completion),
            text=f"Processing... Completion: {percentage_completion:.2f}%",
        )
        self.ui_update_in_progress = False

    async def process_urls(self, progress_bar: DeltaGenerator, table_name: str):
        """
        Asynchronously processes a list of URLs.
        Parameters:
            progress_bar (DeltaGenerator): Streamlit UI element for progress indication.
            table_name (str): The name of the table where data is to be stored.
        """

        timeout = aiohttp.ClientTimeout(
            total=30
        )  # 10 seconds timeout for the entire request process
        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:  # If needed, use session for HTTP requests
            tasks = [
                self.handle_url(url, progress_bar, session, table_name)
                for url in self.urls
            ]
            await asyncio.gather(*tasks)

    def async_crawl_and_ingest(
        self, sitemap_url: str, progress_bar: DeltaGenerator, table_name: str
    ):
        """
        Asynchronously crawls and ingests data from a given sitemap URL.
        Parameters:
            sitemap_url (str): The sitemap URL to crawl.
            progress_bar (DeltaGenerator): Streamlit UI element for progress indication.
            table_name (str): The name of the table where data is to be stored.
        """
        if self.urls is None:
            # TODO: Need to cache the following step:
            self.urls = self.get_sitemap_urls(sitemap_url, onlyEnglish=True)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_urls(progress_bar, table_name))
        finally:
            loop.close()

    def async_crawl_and_ingest_list(
        self, sitemap_url_list: List[str], progress_bar: DeltaGenerator, table_name: str
    ):
        """
        Asynchronously crawls and ingests data from a list of sitemap URLs.
        Parameters:
            sitemap_url_list (List[str]): A list of sitemap URLs to crawl.
            progress_bar (DeltaGenerator): Streamlit UI element for progress indication.
            table_name (str): The name of the table where data is to be stored.
        """
        if self.urls is None:
            # TODO: Need to cache the following step:
            all_urls = [
                url
                for sitemap in sitemap_url_list
                for url in self.get_sitemap_urls(sitemap, onlyEnglish=True)
                if "espanol" not in url
            ]
            length_of_urls = len(all_urls)
            unique_urls = list(set(all_urls))
            logging.info(f"Length of all_urls is: {length_of_urls}")
            print(f"Length of all_urls is: {length_of_urls}")
            length_of_unique_urls = len(unique_urls)
            print(f"Length of unique all_urls is: {length_of_unique_urls}")
            logging.info(f"Length of unique all_urls is: {length_of_unique_urls}")
            self.urls = unique_urls

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_urls(progress_bar, table_name))
        finally:
            loop.close()
