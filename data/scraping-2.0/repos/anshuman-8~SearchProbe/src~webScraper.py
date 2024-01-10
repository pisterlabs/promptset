import asyncio
import json
import time
import logging as log
from typing import Iterator, List
from playwright.async_api import async_playwright
from langchain.docstore.document import Document
from src.utils import document2map

LOG_FILES = False

class AsyncChromiumLoader:
    def __init__(self, web_links: List[str]):
        self.web_links = web_links

    async def scrape_browser(self, web_links: List[str]) -> List[Document]:
        """
        Scrape the urls by creating async tasks for each url
        """
        log.info("Starting scraping...")
        results = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            scraping_tasks = [self.scrape_url(browser, web_link) for web_link in web_links]
            results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
            await browser.close()
            log.debug(f"Browser closed")
        return results

    async def scrape_url(self, browser, web_link: str) -> Document:
        """
        Scrape the url and return the document, it also ignores assets
        """
        web_content = ""
        metadata = {"website": web_link['link'],"source": web_link['source'], "title": web_link['title']}
        url = web_link['link']
        log.info(f"Scraping {url}...")
        t_start = time.time()
        try:
            page = await browser.new_page()
            excluded_resource_types = ["stylesheet", "script", "image", "font", "media"]

            async def route_handler(route):
                resource_type = route.request.resource_type
                if resource_type in excluded_resource_types:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route(
                "**/*",
                route_handler
                # lambda route: route.abort()
                # if route.request.resource_type in excluded_resource_types
                # else route.continue_(),
            )
            await page.goto(url, timeout=8000)
            web_content = await page.content()
            t_end = time.time()
            log.info(f"Content scraped for {url} in {t_end - t_start} seconds")
        except Exception as e:
            log.error(f"Error scraping {url}: {e}")
        finally:
            await page.close()
        result_doc = Document(page_content=web_content, metadata=metadata)
        return result_doc

    async def load_data(self) -> List[Document]:
        """
        Load the data from the urls asynchronously
        """
        data = await self.scrape_browser(self.web_links)
        return data
    
async def scrape_with_playwright(results: List[str]) -> List[dict]:
    """
    Scrape the websites using playwright and chunk the text tokens
    """
    t_flag1 = time.time()
    loader = AsyncChromiumLoader(results)
    docs = await loader.load_data()
    t_flag2 = time.time()

    if LOG_FILES:
        with open("src/log_data/docs.json", "w") as f:
            json.dump(document2map(docs), f)

    log.info(f"AsyncChromiumLoader time: { t_flag2 - t_flag1}")

    return docs
