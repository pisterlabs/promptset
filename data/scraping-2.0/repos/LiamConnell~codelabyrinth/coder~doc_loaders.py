import glob
import os
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import logging

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from tqdm import tqdm

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USER = "liamconnell"

LOGGER = logging.getLogger(__name__)

__all__ = ["load_code_files", "load_docs_website", "load_github_repo"]


def load_code_files(directory: str = "./lliam") -> list[Document]:
    code_files = glob.glob(f"{directory}/**/*.py", recursive=True)
    code_files += glob.glob(f"{directory}/**/*.ts", recursive=True)
    code_files += glob.glob(f"{directory}/**/*.tsx", recursive=True)
    code_data = []
    for code_file in code_files:
        with open(code_file, "r", encoding="utf-8") as f:
            code_data.append(
                Document(page_content=f.read(), metadata={"directory": directory, "source": code_file})
            )
    return code_data


def load_docs_website(base_url: str, max_workers: int = 10) -> list[Document]:
    """
    Loads documents from a website by scraping the provided base_url.

    Args:
        base_url (str): The base URL of the website.
        max_workers (int, optional): The maximum number of concurrent workers. Defaults to 10.

    Returns:
        List[Document]: A list of Document objects containing scraped information.

    """
    if base_url[-1] != "/":
        base_url += "/"

    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = [l.get('href') for l in soup.find_all('a', class_="reference internal")]
    links = list(set(links))

    def scrape_doc_(link):
        try:
            doc = _scrape_doc(urllib.parse.urljoin(base_url, link))
        except Exception as e:
            LOGGER.error(e)
            LOGGER.error(traceback.format_exc())
            return
        return Document(
            page_content=doc,
            metadata={"base_url": base_url, "link": link, "source": f"{base_url}{link}"}
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        docs = list(tqdm(executor.map(scrape_doc_, links)))

    docs = [doc for doc in docs if doc is not None]

    return docs


def load_gee_website(base_url: str = "https://developers.google.com", max_workers: int = 10) -> list[Document]:
    def get_links(base_url_):
        response = requests.get(base_url_)
        soup = BeautifulSoup(response.text, 'html.parser')
        x = soup.find("div", class_="devsite-book-nav-wrapper")
        links = [l.get('href') for l in x.find_all('a', class_="devsite-nav-title")]
        return list(set(links))

    links = get_links("https://developers.google.com/earth-engine/guides") + get_links(
        "https://developers.google.com/earth-engine/apidocs")
    links = list(set(links))

    def scrape_doc_(link):
        try:
            doc = _scrape_doc(urllib.parse.urljoin(base_url, link))
        except Exception as e:
            LOGGER.error(e)
            LOGGER.error(traceback.format_exc())
            return
        return Document(
            page_content=doc,
            metadata={"base_url": base_url, "link": link, "source": urllib.parse.urljoin(base_url, link)}
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        docs = list(tqdm(executor.map(scrape_doc_, links)))

    docs = [doc for doc in docs if doc is not None]

    return docs


def _scrape_doc(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find("main", {"id": "main-content"})
    if text is None:
        text = soup.find("div", {"role": "main"})
    if text is None:
        text = soup.find('article', {'role': 'main'})
    if text is None:
        text = soup.find("div", class_="devsite-article-body")

    if soup is None:
        print(f"Nothing found for {url}")
        return

    # remove all script and style elements
    # for script in soup(["script", "style", "nav"]):
    #     script.extract()

    # get text
    text = text.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def load_github_repo(owner: str, repo: str, path: str = "", file_types: list[str] = None) -> list[Document]:
    """
    Loads files from a GitHub repository.

    Args:
        owner (str): The owner of the GitHub repository.
        repo (str): The name of the GitHub repository.
        path (str, optional): The directory path within the repository. Defaults to the root directory.
        file_types (List[str], optional): A list of file types to filter. Defaults to all file types.

    Returns:
        List[Document]: A list of Document objects containing the content of the files.
    """
    file_types = file_types or [""]
    url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, auth=(GITHUB_USER, GITHUB_TOKEN))

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch files from GitHub repository {owner}/{repo}: {response.text}")

    files = response.json()
    docs = []

    for file in files:
        if file["type"] == "file" and any(file["name"].endswith(ft) for ft in file_types):
            content_response = requests.get(file["download_url"], auth=(GITHUB_USER, GITHUB_TOKEN))
            content = content_response.text

            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"owner": owner, "repo": repo, "path": file["path"], "source": file["html_url"]}
                    )
                )
        if file["type"] == "dir":
            docs += load_github_repo(owner, repo, os.path.join(path, file['name']), file_types)

    return docs
