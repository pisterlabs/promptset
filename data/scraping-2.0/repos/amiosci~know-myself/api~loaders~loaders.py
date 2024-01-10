import abc
import re
import tempfile

from urllib.parse import urlparse, ParseResult
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import ArxivLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import HNLoader
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParserLocal


def _get_url_extension(url: ParseResult) -> str:
    return url.path.split(".")[-1]


class DocumentLoader(abc.ABC):
    url: ParseResult

    def __init__(self, url: ParseResult):
        self.url = url

    @staticmethod
    @abc.abstractmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def can_load(url: ParseResult) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self) -> list[Document]:
        raise NotImplementedError()

    @property
    def target_url(self) -> str:
        return self.url.geturl()


class DefaultDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        return True

    def load(self) -> list[Document]:
        return []


class HackerNewsDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        if HackerNewsDocumentLoader.can_load(url):
            id_match = re.match(r"^id=([0-9]+)$", url.query)
            if id_match:
                id = id_match.group()
            else:
                raise RuntimeError("Cannot be loadable without id param")
            return {
                "loader": "hacker_news",
                "id": id,
            }

        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        # https://news.ycombinator.com/item?id=34817881
        valid_site = url.netloc in ["news.ycombinator.com"]
        has_valid_path = url.path == "/item"
        has_id = re.match(r"^id=[0-9]+$", url.query) is not None

        return valid_site and has_valid_path and has_id

    def load(self) -> list[Document]:
        loader = HNLoader(self.target_url)
        return loader.load()


class WikipediaDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        if WikipediaDocumentLoader.can_load(url):
            query = url.path.split("/")[-1]
            return {
                "loader": "wikipedia",
                "id": query,
            }

        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        valid_site = url.netloc in ["wikipedia.org", "en.wikipedia.org"]
        valid_path_prefixes = ["/wiki/"]
        has_valid_path_prefix = any(url.path.startswith(x) for x in valid_path_prefixes)

        return valid_site and has_valid_path_prefix

    def load(self) -> list[Document]:
        query = self._load_query()
        if not query:
            raise ValueError("Invalid Wikipedia URL requested")

        loader = WikipediaLoader(query=query)
        return loader.load()

    def _load_query(self) -> str:
        # https://en.wikipedia.org/wiki/Walt_Disney
        return self.url.path.split("/")[-1]


def _load_arxiv_query(url: ParseResult) -> str:
    if url.path.startswith("/pdf/"):
        # https://arxiv.org/pdf/2305.05003.pdf
        return url.path[1:].strip("pdf")[1:-1]

    # https://arxiv.org/abs/2305.05003
    return url.path.split("/")[-1]


class ArxivDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        if WikipediaDocumentLoader.can_load(url):
            return {
                "loader": "arxiv",
                "id": _load_arxiv_query(url),
            }

        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        valid_site = url.netloc == "arxiv.org"
        valid_path_prefixes = ["/pdf/", "/abs/"]
        has_valid_path_prefix = any(url.path.startswith(x) for x in valid_path_prefixes)

        return valid_site and has_valid_path_prefix

    def load(self) -> list[Document]:
        query = _load_arxiv_query(self.url)
        if not query:
            raise ValueError("Invalid Arxiv URL requested")

        loader = ArxivLoader(query=query)
        return loader.load()


class PDFDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        if PDFDocumentLoader.can_load(url):
            return {
                "loader": "pdf",
                "url": url.geturl(),
            }

        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        extension = _get_url_extension(url)
        return extension.lower() in ["pdf"]

    def load(self) -> list[Document]:
        loader = PyPDFLoader(self.target_url)
        pages = loader.load_and_split()
        return pages


class WebPageDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        if WebPageDocumentLoader.can_load(url):
            return {
                "loader": "webpage",
                "url": url.geturl(),
            }

        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        valid_schemes = ["http", "https"]
        return url.scheme in valid_schemes

    def load(self) -> list[Document]:
        loader = SeleniumURLLoader(urls=[self.target_url])
        text = loader.load()

        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(text)

        text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=300)
        texts = text_splitter.split_documents(docs_transformed)

        return texts


class YouTubeVideoDocumentLoader(DocumentLoader):
    @staticmethod
    def get_loader_spec(url: ParseResult) -> dict[str, str] | None:
        if YouTubeVideoDocumentLoader.can_load(url):
            return {
                "loader": "youtube",
                "url": url.geturl(),
            }

        return None

    @staticmethod
    def can_load(url: ParseResult) -> bool:
        is_youtube = url.netloc in ["www.youtube.com", "youtube.com"]
        valid_schemes = ["http", "https"]
        return (url.scheme in valid_schemes) and is_youtube

    def load(self) -> list[Document]:
        with tempfile.TemporaryDirectory() as d:
            loader = GenericLoader(
                YoutubeAudioLoader([self.target_url], d), OpenAIWhisperParserLocal()
            )
            return loader.load()


# TODO: Define by namespace lookup
_ORDERED_LOADERS = [
    # site specific
    ArxivDocumentLoader,
    WikipediaDocumentLoader,
    HackerNewsDocumentLoader,
    YouTubeVideoDocumentLoader,
    # file specific
    PDFDocumentLoader,
    # Fallback
    WebPageDocumentLoader,
    DefaultDocumentLoader,
]


def get_loader_spec(url: str) -> dict[str, str] | None:
    parsed_url = urlparse(url)
    return next(
        (
            x.get_loader_spec(parsed_url)
            for x in _ORDERED_LOADERS
            if x.get_loader_spec(parsed_url) is not None
        ),
        None,
    )


def locate(url: str) -> DocumentLoader | None:
    parsed_url = urlparse(url)
    loader = next(
        (x for x in _ORDERED_LOADERS if x.can_load(parsed_url)),
        None,
    )
    if loader is not None:
        return loader(parsed_url)

    return None


if __name__ == "__main__":

    def confirm_loader_functionality(url: str, loader_type: type):
        loader = locate(url)

        assert loader is not None
        assert isinstance(loader, loader_type)

        docs = loader.load()
        assert len(docs) > 0

    webpage_url = "https://docs.python.org/3/library/urllib.parse.html"
    confirm_loader_functionality(webpage_url, WebPageDocumentLoader)

    arxiv_urls = [
        "https://arxiv.org/pdf/2305.05003.pdf",
        "https://arxiv.org/abs/2305.05003",
    ]
    for arxiv_url in arxiv_urls:
        confirm_loader_functionality(arxiv_url, ArxivDocumentLoader)

    youtube_urls = [
        "https://youtube.com/shorts/IicbiwTAslE?si=H1qA7---M4ZiuHTc",
        # Too large for GPU under standard usage
        # Evaluate PowerInfer, and the like
        # "https://www.youtube.com/watch?v=edyqWHRgSX8",
    ]

    for youtube_url in youtube_urls:
        confirm_loader_functionality(youtube_url, YouTubeVideoDocumentLoader)

    wikipedia_url = "https://en.wikipedia.org/wiki/Walt_Disney"
    confirm_loader_functionality(wikipedia_url, WikipediaDocumentLoader)

    hn_url = "https://news.ycombinator.com/item?id=34817881"
    confirm_loader_functionality(hn_url, HackerNewsDocumentLoader)
