from __future__ import annotations

from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from .unit_of_work import VectorStoreUnitOfWork


SYSTEM_TEMPLATE = """Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""

MESSAGES = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]
PROMPT = ChatPromptTemplate.from_messages(MESSAGES)


def query(question: str, uow: VectorStoreUnitOfWork) -> str:
    chain_type_kwargs = {"prompt": PROMPT}
    with uow:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            retriever=uow.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        result = chain(question)
        return result["answer"]


def split(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def add_to_store(documents: List[Document], unit_of_work: VectorStoreUnitOfWork):
    with unit_of_work:
        unit_of_work.vector_store.add_documents(documents)
        unit_of_work.persist()


# pylint: disable=unused-argument, unused-variable
def crawl(base_url: str, unit_of_work: VectorStoreUnitOfWork, **kwargs):
    # options
    timeout = kwargs.pop("timeout", 60)
    limit = kwargs.pop("limit")

    base_domain = urlparse(base_url).netloc
    visited_urls = set()
    urls_to_visit = set()
    urls_to_ignore = set()

    urls_to_visit.add(base_url)

    def visit_url(url):
        visited_urls.add(url)
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            for anchor in soup.find_all("a"):
                link_url = anchor.get("href")
                if link_url is not None and link_url != "":
                    link_url = urljoin(url, link_url)
                    # Remove internal anchor links:
                    if "#" in link_url:
                        link_url = link_url[: link_url.index("#")]
                    # Check the domain is same as original domain
                    if urlparse(link_url).netloc == base_domain:
                        if (
                            link_url not in visited_urls
                            and link_url not in urls_to_ignore
                        ):
                            urls_to_visit.add(link_url)
        else:
            # logging.warning(f"Error opening {url} : {response.status_code}")
            # We should stop trying to open this URL
            urls_to_ignore.add(url)

    while urls_to_visit:
        url = urls_to_visit.pop()
        yield url
        try:
            visit_url(url)
        except requests.exceptions.RequestException as exc:
            print("Error: " + str(exc))


def load_documents(url) -> List[Document]:
    loader = WebBaseLoader(url)
    return loader.load()


def scrap(url: str, uow: VectorStoreUnitOfWork):
    docs = load_documents(url)
    splits = split(docs)
    add_to_store(splits, uow)
