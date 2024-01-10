import logging
import typing as t

import requests_cache

from pueblo.context import pueblo_cache_path

if t.TYPE_CHECKING:
    from langchain.schema import Document

http_cache_file = pueblo_cache_path / ".httpcache.sqlite"
http = requests_cache.CachedSession(str(http_cache_file))

logger = logging.getLogger(__name__)


class CachedWebResource:
    """
    A basic wrapper around `requests-cache` and `langchain`.
    """

    def __init__(self, url: str):
        logger.info(f"Using web cache: {http_cache_file}")
        self.url = url

    def fetch_single(self) -> t.List["Document"]:
        return [self.document_from_url()]

    @staticmethod
    def fetch_multi(urls) -> t.List["Document"]:
        from langchain.document_loaders import UnstructuredURLLoader

        loader = UnstructuredURLLoader(urls=urls)
        return loader.load()

    def document_from_url(self) -> "Document":
        """
        Converge URL resource into LangChain Document.
        """
        logger.info(f"Acquiring web resource: {self.url}")
        from langchain.schema import Document
        from unstructured.partition.html import partition_html

        response = http.get(self.url)
        elements = partition_html(text=response.text)
        text = "\n\n".join([str(el) for el in elements])
        metadata = {"source": self.url}
        return Document(page_content=text, metadata=metadata)

    def langchain_documents(self, **kwargs) -> t.List["Document"]:
        """
        Load URL resource, and split paragraphs in response into individual documents.
        """
        from langchain.text_splitter import CharacterTextSplitter

        documents = self.fetch_single()
        text_splitter = CharacterTextSplitter(**kwargs)
        return text_splitter.split_documents(documents)


if __name__ == "__main__":
    from pueblo import setup_logging

    setup_logging()
    url = "https://github.com/langchain-ai/langchain/raw/v0.0.325/docs/docs/modules/state_of_the_union.txt"
    docs = CachedWebResource(url).langchain_documents(chunk_size=1000, chunk_overlap=0)
    print("docs:", docs)  # noqa: T201
