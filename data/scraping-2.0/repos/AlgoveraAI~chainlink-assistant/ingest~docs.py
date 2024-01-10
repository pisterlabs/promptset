# Required libraries
import re
import os
import bs4
import time
import pickle
import requests
import html2text
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Optional, Set
from requests.exceptions import RequestException
from langchain.docstore.document import Document

from config import DATA_DIR, get_logger, MAX_THREADS
from ingest.utils import (
    remove_prefix_text,
    extract_first_n_paragraphs,
    get_description_chain,
    get_driver,
)

logger = get_logger(__name__)

# Settings for requests
REQUEST_DELAY = 0.1
SESSION = requests.Session()

# Get the driver
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


def fetch_url_selenium(url: str):
    local_driver = get_driver()
    try:
        local_driver.get(url)
        local_driver.implicitly_wait(3)
        time.sleep(3)
        source = local_driver.page_source
    except RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        source = None
    finally:
        local_driver.quit()
    return source


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


def process_tag(tag: bs4.element.Tag):
    """
    Process an HTML tag. If the tag is a table, convert it to Markdown.
    Otherwise, convert it to Markdown as-is.

    :param tag: HTML tag to process.
    :return: Markdown representation of the tag.
    """
    if tag.name == "table":
        # Convert the table to a DataFrame
        df = pd.read_html(str(tag))[0]

        # Convert the DataFrame to Markdown
        return df.to_markdown(index=False) + "\n"
    else:
        # If it's not a table, convert it to Markdown as before
        html = str(tag)
        return html2text.html2text(html)


def fix_markdown_links(markdown_text: str):
    """
    Fix Markdown links by removing any spaces in the URL.

    :param markdown_text: Markdown text to process.
    :return: Fixed Markdown text.
    """
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\s+([^)]+)\)", r"[\1](\2\3)", markdown_text)


def process_nested_tags(tag: bs4.element.Tag):
    """
    Process nested HTML tags. Convert tags to Markdown recursively.

    :param tag: Root HTML tag to process.
    :return: Markdown text
    """
    if tag.name in {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "pre",
        "table",
        "ol",
        "ul",
    }:
        return process_tag(tag)
    else:
        markdown_parts = []
        for child in tag.children:
            if isinstance(child, bs4.element.Tag):
                markdown_parts.append(process_nested_tags(child))
        return "".join(markdown_parts)


# def parse(url, use_selenium:list=[]):
#     if url in use_selenium:
#         response = fetch_url_selenium(url)
#         if response:
#             soup = BeautifulSoup(response, 'html.parser')
#         else:
#             soup = None
#     else:
#         response = fetch_url_request(url)
#         if response:
#             soup = BeautifulSoup(response.content, 'html.parser')
#         else:
#             soup = None
#     if soup:
#         return parse_from_soup(soup)


def parse(url: str):
    """
    Fetches and parses a URL using Selenium and BeautifulSoup.
    Extracts the useful information from the HTML and returns it.

    :param url: URL to fetch and parse.
    :return: Processed content from the URL if it exists, None otherwise.
    """
    # Fetch the page with Selenium
    html = fetch_url_selenium(url)

    if html:
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

    # Continue processing the page as before
    if soup:
        return parse_from_soup(soup)


def parse_from_soup(soup: bs4.BeautifulSoup):
    """
    Parses the soup object from BeautifulSoup, removes unnecessary tags,
    and returns the content in markdown format.

    :param soup: BeautifulSoup object
    :return: Content from the soup in markdown format.
    """
    grid_main = soup.find("div", {"id": "grid-main"})

    if grid_main:
        for img in grid_main.find_all("img"):
            img.decompose()

        for h2 in grid_main.find_all("h2", {"class": "heading"}):
            h2.decompose()

        markdown_content = process_nested_tags(grid_main)
        fixed_markdown_content = fix_markdown_links(markdown_content)
        return fixed_markdown_content
    else:
        logger.error('Failed to find the "grid-main" div.')


def remove_duplicates(doc_list: List[Document]):
    """
    Removes duplicate documents from a list of Documents based on page_content.

    :param doc_list: List of Document objects.
    :return: List of unique Document objects.
    """
    content_to_doc = {}
    for doc in doc_list:
        if doc.page_content not in content_to_doc:
            content_to_doc[doc.page_content] = doc
    return list(content_to_doc.values())


def insert_full_url(text: str):
    """
    Inserts the full URL into Markdown links in the text.

    :param text: Text to process.
    :return: Text with full URLs in Markdown links.
    """
    base_url = "https://docs.chain.link"

    def replacer(match):
        sub_url = match.group(2)
        # If the sub_url is an absolute URL, return it unchanged
        if sub_url.startswith("http://") or sub_url.startswith("https://"):
            return match.group(0)
        # If the sub_url starts with a slash, remove it to avoid double slashes in the final url
        if sub_url.startswith("/"):
            sub_url = sub_url[1:]
        return f"[{match.group(1)}]({base_url}/{sub_url})"

    return re.sub(r"\[(.*?)\]\((.*?)\)", replacer, text)


def refine_docs(docs: List[Document]):
    """
    Removes duplicates and inserts full URLs into the page_content of the Document objects.

    :param docs: List of Document objects.
    :return: Refined list of Document objects.
    """
    docs_filtered = remove_duplicates(docs)
    base_url = "https://docs.chain.link"
    for doc in docs_filtered:
        doc.page_content = insert_full_url(doc.page_content)
    return docs_filtered


def scrap_docs():
    global driver
    driver = get_driver()

    all_urls = get_all_suburls("https://docs.chain.link/")
    all_urls = sorted(list(set(all_urls)))

    # Get description chain
    chain_description = get_description_chain()

    docs_documents = []

    # Utilizing ThreadPoolExecutor to parallelize the fetching and processing of URLs
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_url = {executor.submit(parse, url): url for url in all_urls}
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url), total=len(all_urls)
        ):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as e:
                logger.error(f"Exception occurred in scraping {url}: {e}")
                continue

            if data:
                # Remove anything above title
                markdown_content = remove_prefix_text(data)
                # Get title
                try:
                    titles = re.findall(r"^#\s(.+)$", markdown_content, re.MULTILINE)
                    title = titles[0].strip()
                except:
                    title = markdown_content.split("\n\n")[0].replace("#", "").strip()

                # Get description
                para = extract_first_n_paragraphs(markdown_content, num_para=2)
                description = chain_description.predict(context=para)

                docs_documents.append(
                    Document(
                        page_content=data,
                        metadata={
                            "source": url,
                            "source_type": "technical_document",
                            "title": title,
                            "description": description,
                        },
                    )
                )

    docs_documents = remove_duplicates(docs_documents)

    # Save the documents to a pickle file with date in the name
    with open(f"{DATA_DIR}/tech_documents.pkl", "wb") as f:
        pickle.dump(docs_documents, f)

    logger.info(f"Scraped technical documents.")

    return docs_documents
