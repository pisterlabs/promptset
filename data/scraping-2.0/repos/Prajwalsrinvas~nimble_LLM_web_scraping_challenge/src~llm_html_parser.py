import os
from urllib.parse import urljoin

from langchain.chains.openai_functions import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from loguru import logger
from openai.error import InvalidRequestError
from pydantic import BaseModel, Field
from selectolax.lexbor import LexborHTMLParser

from config.config import config, secrets
from src.nimble import hit_nimble_api
from src.preprocess_html import _process_html
from src.utils import _convert_url_to_file_title


class LLMItem(BaseModel):
    product_url_css_selector: str = Field(
        description="return the CSS selector that extracts all product links from given HTML. The product link refers to the main entity on a page. Ex: The product can be an item on a ecommerce platform or a movie on imdb. The selector that you write should directly point to the link and not any other nodes."
    )
    pagination_url_css_selector: str = Field(
        description="return the CSS selector that extracts all pagination links from given HTML. The pagination links refers to links that lead to the next page from the current page. They are enumerated and often have a pattern like: page=1, page=2 etc. The selector that you write should directly point to the link and not any other nodes. The selector should not consider the next page and previous page links"
    )


def hit_openai_api(html):
    llm = ChatOpenAI(
        temperature=0,
        model=config["openai"]["models"][0],
        openai_api_key=secrets["openai"]["api_key"],
    )
    chain = create_extraction_chain_pydantic(pydantic_schema=LLMItem, llm=llm)
    # loader = TextLoader(
    #     file_path=html,
    #     encoding="utf-8",
    # )
    # data = loader.load()

    llm_item = chain.run(html)
    return llm_item[0].dict()


def _extract_links(tree, selector, base_url):
    urls = [node.attributes.get("href") for node in tree.css(selector)]
    urls = [urljoin(base_url, relative_url) for relative_url in urls]
    return urls


def extract_elements(base_url, html, selectors):
    tree = LexborHTMLParser(html)
    product_urls = _extract_links(tree, selectors["product_url_css_selector"], base_url)
    pagination_urls = _extract_links(
        tree, selectors["pagination_url_css_selector"], base_url
    )
    return product_urls, pagination_urls


def get_selectors(url, store_html):
    html = hit_nimble_api(url)
    clean_html = _process_html(html)
    if store_html:
        BASE_HTML_FOLDER = "html"
        os.makedirs(BASE_HTML_FOLDER, exist_ok=True)
        raw_html_file_path = os.path.join(
            BASE_HTML_FOLDER, f"{_convert_url_to_file_title(url)}.html"
        )
        with open(raw_html_file_path, "w", encoding="utf-8") as file:
            file.write(html)

        clean_html_file_path = os.path.join(
            BASE_HTML_FOLDER, f"{_convert_url_to_file_title(url)}_clean.html"
        )
        with open(clean_html_file_path, "w", encoding="utf-8") as file:
            file.write(clean_html)

    try:
        selectors = hit_openai_api(clean_html)
        return selectors, html
    except InvalidRequestError as e:
        logger.error(str(e))
        return None, None
