import re
import os
import time
import json
import pickle
import requests
import html2text
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Optional, Set, Dict, Tuple
from requests.exceptions import RequestException

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver

from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.document_loaders import YoutubeLoader

from config import DATA_DIR, get_logger, MAX_THREADS
from ingest.utils import (
    get_description_chain,
    remove_prefix_text,
    extract_first_n_paragraphs,
    get_driver,
)

logger = get_logger(__name__)

# Settings for requests
REQUEST_DELAY = 0.1
TIMEOUT = 10
SESSION = requests.Session()

driver = None


def filter_urls_by_base_url(urls: List, base_url: str):
    """
    Filters a list of URLs and returns only those that include the base_url.

    :param urls: List of URLs to filter.
    :param base_url: Base URL to filter by.
    :return: List of URLs that include the base_url.
    """
    return [url for url in urls if base_url in url]


def normalize_url(url: str):
    """
    Normalize a URL by ensuring it ends with '/'.

    :param url: URL to normalize.
    :return: Normalized URL.
    """
    return url if url.endswith("/") else url + "/"


def fetch_url_request(url: str):
    """
    Fetches the content of a URL using requests library and returns the response.
    In case of any exception during fetching, logs the error and returns None.

    :param url: URL to fetch.
    :return: Response object on successful fetch, None otherwise.
    """
    try:
        response = SESSION.get(url)
        response.raise_for_status()
        return response
    except RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


# Use Selenium's WebDriverWait instead of time.sleep
def fetch_url_selenium(url: str):
    try:
        driver.get(url)
        WebDriverWait(driver, TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        return driver.page_source

    except RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def process_url(response: requests.Response, visited: Set, base_url: str):
    """
    Process a URL response. Extract all absolute URLs from the response that
    haven't been visited yet and belong to the same base_url.

    :param response: Response object from a URL fetch.
    :param visited: Set of URLs already visited.
    :param base_url: Base URL to filter by.
    :return: Set of new URLs to visit.
    """
    urls = set()
    if response:
        soup = BeautifulSoup(response.content, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            if href is not None and "#" not in href:
                absolute_url = normalize_url(urljoin(response.url, href))
                if absolute_url not in visited and base_url in absolute_url:
                    visited.add(absolute_url)
                    urls.add(absolute_url)
    return urls


def get_all_suburls(url: str, visited: Optional[Set] = None):
    """
    Get all sub-URLs of a given URL that belong to the same domain.

    :param url: Base URL to start the search.
    :param visited: Set of URLs already visited.
    :return: Set of all sub-URLs.
    """
    if visited is None:
        visited = set()

    if not url.startswith("http"):
        url = "https://" + url

    base_url = url.split("//")[1].split("/")[0]
    urls = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_responses = [executor.submit(fetch_url_request, url)]
        while future_responses:
            for future in concurrent.futures.as_completed(future_responses):
                future_responses.remove(future)
                response = future.result()
                new_urls = process_url(response, visited, base_url)
                urls.update(new_urls)
                if len(future_responses) < MAX_THREADS:
                    for new_url in new_urls:
                        future_responses.append(
                            executor.submit(fetch_url_request, new_url)
                        )

    urls = filter_urls_by_base_url(urls, base_url)
    return urls


def is_there_video(soup: BeautifulSoup) -> List[str]:
    """Check if there is a video in the soup
    params:
        soup: BeautifulSoup object
    returns:
        video_links: List of video links
    """
    iframes = soup.find_all("iframe")
    video_links = []
    for iframe in iframes:
        src = iframe.get("src")
        if "youtube" in src:
            video_links.append(src)

    return video_links


def get_youtube_docs(video_tags: List[str], chain_description) -> List[Document]:
    """Get youtube docs from the video tags
    params:
        video_tags: List of video tags
    returns:
        u_tube_docs: List of youtube docs
    """
    if video_tags:
        u_tube_docs = []
        for v_tag in video_tags:
            try:
                u_tube = json.loads(v_tag.script.string)["items"][0]["url"]
                u_tube_id = YoutubeLoader.extract_video_id(u_tube)
                u_tube_doc = YoutubeLoader(u_tube_id, add_video_info=True).load()[0]

                # Get description
                description = chain_description.predict(
                    context=u_tube_doc.page_content[:1500]
                )

                # Make sure its Chainlink video
                assert u_tube_doc.metadata["author"].lower() == "chainlink"
                u_tube_doc.metadata = {
                    "source": u_tube,
                    "source_type": "video",
                    "title": u_tube_doc.metadata["title"],
                    "description": description,
                }

                # Append to the list
                u_tube_docs.append(u_tube_doc)

            except Exception as e:
                print(e)
                u_tube_doc = []
    else:
        u_tube_docs = []
    return u_tube_docs


def scrap_url(
    url: str, chain_description: LLMChain, driver: webdriver.Chrome = driver
) -> Document:
    """Process a URL and return a list of words
    param url: URL to process
    param driver: Selenium driver
    return: Document object
    """
    driver.get(url)
    driver.implicitly_wait(2)
    time.sleep(2)

    # Get the page source
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Get the Markdown content
    # Remove images, videos, SVGs, and other media elements; also nav
    for media_tag in soup.find_all(
        ["img", "video", "svg", "audio", "source", "track", "picture", "nav"]
    ):
        media_tag.decompose()

    # Remove the footer (assuming it's in a <footer> tag or has a class/id like 'footer')
    for footer_tag in soup.find_all(["footer", {"class": "footer"}, {"id": "footer"}]):
        footer_tag.decompose()

    # Remove sections with class="section-page-alert"
    for page_alert in soup.find_all("div", class_="section-page-alert"):
        page_alert.decompose()

    # Remove sections with class="cta-subscribe"
    for cta_subscribe in soup.find_all(class_="cta-subscribe"):
        cta_subscribe.decompose()

    html_content = str(soup)
    h = html2text.HTML2Text()
    markdown_content = h.handle(html_content)

    # Remove the prefix
    markdown_content = remove_prefix_text(markdown_content)

    # Get the title
    titles = re.findall(r"^#\s(.+)$", markdown_content, re.MULTILINE)
    title = titles[0].strip()

    # Get description
    para = extract_first_n_paragraphs(markdown_content, num_para=2)
    description = chain_description.predict(context=para)

    # Put the markdown content into a Document object
    doc = Document(
        page_content=markdown_content,
        metadata={
            "source": url,
            "title": title,
            "description": description,
            "source_type": "main",
        },
    )

    # Get YouTube docs
    video_tags = soup.find_all("a", href=True, class_="techtalk-video-lightbox")
    u_tube_docs = get_youtube_docs(video_tags, chain_description)

    return doc, u_tube_docs


def concurrent_fetch_url_selenium(url: str):
    driver = get_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        source = driver.page_source
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        source = None
    driver.quit()
    return source


def concurrent_scrap_url(url: str, chain_description: LLMChain):
    try:
        logger.info(f"Processing {url}")
        page_source = concurrent_fetch_url_selenium(url)
        if page_source is None:
            return None, []

        soup = BeautifulSoup(page_source, "html.parser")
        # Get the Markdown content
        # Remove images, videos, SVGs, and other media elements; also nav
        for media_tag in soup.find_all(
            ["img", "video", "svg", "audio", "source", "track", "picture", "nav"]
        ):
            media_tag.decompose()

        # Remove the footer (assuming it's in a <footer> tag or has a class/id like 'footer')
        for footer_tag in soup.find_all(
            ["footer", {"class": "footer"}, {"id": "footer"}]
        ):
            footer_tag.decompose()

        # Remove sections with class="section-page-alert"
        for page_alert in soup.find_all("div", class_="section-page-alert"):
            page_alert.decompose()

        # Remove sections with class="cta-subscribe"
        for cta_subscribe in soup.find_all(class_="cta-subscribe"):
            cta_subscribe.decompose()

        html_content = str(soup)
        h = html2text.HTML2Text()
        markdown_content = h.handle(html_content)

        # Remove the prefix
        markdown_content = remove_prefix_text(markdown_content)

        # Get the title
        titles = re.findall(r"^#\s(.+)$", markdown_content, re.MULTILINE)
        title = titles[0].strip()

        # Get description
        para = extract_first_n_paragraphs(markdown_content, num_para=2)
        description = chain_description.predict(context=para)

        # Put the markdown content into a Document object
        doc = Document(
            page_content=markdown_content,
            metadata={
                "source": url,
                "title": title,
                "description": description,
                "source_type": "main",
            },
        )

        # Get YouTube docs
        video_tags = soup.find_all("a", href=True, class_="techtalk-video-lightbox")
        u_tube_docs = get_youtube_docs(video_tags, chain_description)

        return doc, u_tube_docs
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return None, []


def scrap_chain_link() -> Tuple[List[Dict], List[Dict]]:
    """
    Scrap all the urls from https://chain.link/ and save the main docs and you tube docs to disk
    return: Tuple[List[Dict], List[Dict]]
    """
    global driver
    driver = get_driver()

    raw_urls = get_all_suburls("https://chain.link/")
    raw_urls = list(
        set([url for url in raw_urls if url.startswith("https://chain.link")])
    )
    if "https://chain.link/faqs" not in raw_urls:
        raw_urls.append("https://chain.link/faqs")

    all_main_docs = []
    all_you_tube_docs = []
    chain_description = get_description_chain()

    progress_bar = tqdm(total=len(raw_urls), desc="Processing URLs", position=0, leave=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_url = {
            executor.submit(concurrent_scrap_url, url, chain_description): url
            for url in raw_urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                main_doc, you_tube_docs = future.result()
                if main_doc:
                    all_main_docs.append(main_doc)
                if you_tube_docs:
                    all_you_tube_docs.extend(you_tube_docs)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
            
            # Update the tqdm progress bar
            progress_bar.update(1)

    # remove deplicates
    all_main_docs = list({doc.metadata["source"]: doc for doc in all_main_docs}.values())
    all_you_tube_docs = list(
        {doc.metadata["source"]: doc for doc in all_you_tube_docs}.values()
    )

    # remove https://chain.link/terms
    all_main_docs = [
        doc for doc in all_main_docs if doc.metadata["source"] != "https://chain.link/terms"
    ]

    # Save to disk as pickle
    with open(f"{DATA_DIR}/chain_link_main_documents.pkl", "wb") as f:
        pickle.dump(all_main_docs, f)

    with open(f"{DATA_DIR}/chain_link_you_tube_documents.pkl", "wb") as f:
        pickle.dump(all_you_tube_docs, f)

    logger.info("Done")

    return all_main_docs, all_you_tube_docs
