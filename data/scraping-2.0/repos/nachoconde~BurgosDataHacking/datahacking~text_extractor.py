import requests
from utils import (
    fetch_page,
    parse_html,
    extract_link_and_text,
    preprocess_text,
    dump_jsonl,
    load_config,
)
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from langchain.document_loaders import PyPDFLoader
import constants as constants
from constants import AGENT as AGENT


def download_news_BurgosConecta(
    link: str, name: str, title: str, jsonl_file_name: str
) -> None:
    """ """
    page = requests.get(link, headers=AGENT)
    soup = BeautifulSoup(page.text, "lxml")

    div_content = soup.find("div", class_="v-d v-d--ab-c v-d--bs")

    if div_content is not None:
        paragraphs = div_content.find_all("p", class_="v-p")
        text = ""
        for paragraph in paragraphs:
            if (
                "Copiar enlace" in paragraph.text
                or "WhatsApp" in paragraph.text
                or "Facebook" in paragraph.text
                or "X" in paragraph.text
                or "LinkedIn" in paragraph.text
                or "Telegram" in paragraph.text
            ):
                continue
            text += paragraph.text + " "

        text = preprocess_text(text)
        title = preprocess_text(title)

        file_dict = {
            "src": name,
            "titulo": title,
            "texto": text,
        }
        print("Saving dict in JSONL: " + (title))

        dump_jsonl(jsonl_file_name, file_dict)
    else:
        print("No text found on: " + link)


def find_links_rss(
    url: str,
    name: str,
    jsonl_file_name: str,
    rss_file_name: str,
    src: str,
) -> None:
    """
    Process a RSS file and call a function that extracts the post in JSONL
    """
    response = requests.get(url, headers=constants.AGENT)

    # save rss file locally
    with open(rss_file_name, "w") as f:
        f.write(response.text)

    # extract all links from rss file
    tree = ET.parse(rss_file_name)
    root = tree.getroot()
    items = root.findall(".//item")

    for item in items:
        # Get the link & title from the <link> sub-element
        link = item.find("link")
        title = item.find("title")
        if src == "BurgosConecta":
            download_news_BurgosConecta(link.text, name, title.text, jsonl_file_name)


def download_news(
    link: str, name: str, title: str, jsonl_file_name: str, paragraph_class
) -> None:
    """
    Fetches a link from a page and extracts the text.
    Then it stores it in a jsonl file.
    #PoloPositivo -> no class
    #DiarioDeBurgos -> class_="v-p"

    """

    page = fetch_page(link)
    if page is not None:
        soup = parse_html(page)
    if soup is None:
        print("Error: Unable to parse HTML")
        return

    paragraphs = soup.find_all("p", class_=paragraph_class)

    text = " ".join(paragraph.text for paragraph in paragraphs)
    text = preprocess_text(text)
    title = preprocess_text(title)

    file_dict = {
        "src": name,
        "titulo": title,
        "texto": text,
    }
    print("Saving link to JSONL: " + title)

    dump_jsonl(jsonl_file_name, file_dict)


def find_articles(
    url: str,
    name: str,
    jsonl_file_name: str,
    src: str,
    paragraph_class: str = None,
) -> None:
    """
    Find html tag articles and call a function that extracts the post in JSONL
    """
    print("Processing page: " + url)
    page = fetch_page(url)
    soup = None
    if page is not None:
        soup = parse_html(page)

    if soup is not None:
        if (
            src == "DiarioDeBurgos"
            or src == "ElCorreodeBurgos"
            or src == "BurgosConecta"
        ):
            articles = []
            articles = soup.find_all("article")

            for item in articles:
                link, title = extract_link_and_text(item)
                if link is None or title is None:
                    continue
                if src == "DiarioDeBurgos":
                    link = "https://www.diariodeburgos.es/" + link

                download_news(link, name, title, jsonl_file_name, paragraph_class)

        elif src == "PoloPositivo":
            news = soup.find_all("a", class_="text-white entry-title")
            for item in news:
                href = item.get("href")
                download_news(href, name, item.text, jsonl_file_name, paragraph_class)
        else:
            print(f"Error: Unknown type of source: {src}")


def process_pdf(url: str, name: str, jsonl_file_name: str, src: str) -> None:
    """
    Process a pdf file and extract the text.
    """
    response = requests.get(url, headers=constants.AGENT)

    # save pdf file locally
    with open("datahacking/pdfs/" + name, "wb") as f:
        f.write(response.content)

    loader = PyPDFLoader(
        "datahacking/pdfs/" + name,
    )
    docs = loader.load()

    for doc in docs:
        text = preprocess_text(doc.page_content)
        if len(text) > 200:
            file_dict = {
                "src": src,
                "titulo": name,
                "texto": text,
            }
            print("Saving pdf page to JSONL")
            dump_jsonl(jsonl_file_name, file_dict)


def process_config(config_section, process_function):
    if config_section["ingest"]:
        for url in config_section["urls"]:
            if config_section["type"] == "rss":
                print("Processing RSS:" + url["link"])
                find_links_rss(
                    url["link"],
                    url["name"],
                    config_section["jsonl_file"],
                    url["rss_file"],
                    config_section["src"],
                )
            elif config_section["type"] == "pdf":
                print("Processing PDF:" + url["link"])
                process_function(
                    url["link"],
                    url["name"],
                    config_section["jsonl_file"],
                    config_section["src"],
                )
            elif config_section["type"] == "articles":
                for i in range(int(url["paginas"])):
                    link = url["link"] + str(i)
                    if config_section["src"] == "BurgosConecta":
                        link = link + ".html"
                    print(f"Processing link: {link}")

                    process_function(
                        link,
                        url["name"],
                        config_section["jsonl_file"],
                        config_section["src"],
                        config_section["paragraph_class"],
                    )
            elif config_section["type"] == "links":
                pass


def main():
    config = load_config()
    print("Starting ingestion")
    process_config(config["burgosConectaRSS"], find_links_rss)
    process_config(config["PoloPositivo"], find_articles)
    process_config(config["burgosConectaNews"], find_articles)
    process_config(config["DiarioDeBurgos"], find_articles)
    process_config(config["elcorreodeburgos"], find_articles)
    process_config(config["SEPE"], process_pdf)


if __name__ == "__main__":
    main()
