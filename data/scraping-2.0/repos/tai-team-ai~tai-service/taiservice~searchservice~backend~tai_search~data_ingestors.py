"""Define data ingestors used by the tai_search."""
from typing import Optional, Type, Union
from abc import ABC, abstractmethod
from uuid import uuid4
from enum import Enum
from pathlib import Path
import traceback
import urllib.request
import urllib.parse
import filetype
from bs4 import BeautifulSoup
from pydantic import HttpUrl
import tiktoken
from loguru import logger
import requests
from langchain.schema import Document
from langchain.document_loaders.youtube import (
    ALLOWED_NETLOCK as YOUTUBE_NETLOCS,
    YoutubeLoader,
)
from .data_ingestor_schema import (
    IngestedDocument,
    LatexExtension,
    MarkdownExtension,
    InputDocument,
    InputFormat,
    InputDataIngestStrategy,
)


def number_tokens(text: str) -> int:
    """Get the number of tokens in the text."""
    # the cl100k_base is the encoding for chat models
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


class Ingestor(ABC):
    """Define the ingestor class."""

    @classmethod
    @abstractmethod
    def ingest_data(cls, input_data: InputDocument, bucket_name: str) -> IngestedDocument:
        """Ingest the data."""

    @classmethod
    def _get_input_format(cls, input_pointer: str) -> InputFormat:
        """Get the file type."""

        def check_file_type(path: Path, extension_enum: Type[Enum]) -> bool:
            """Check if the file type matches given extensions."""
            return path.suffix in [extension.value for extension in extension_enum]

        def get_text_file_type(path: Path) -> InputFormat:
            """Get the text file type."""
            with open(path, "r", encoding="utf-8") as f:
                file_contents = f.read()
            is_html = bool(BeautifulSoup(file_contents, "html.parser").find())
            if is_html:
                return InputFormat.HTML
            elif check_file_type(path, LatexExtension):
                return InputFormat.LATEX
            elif check_file_type(path, MarkdownExtension):
                return InputFormat.MARKDOWN
            return InputFormat.GENERIC_TEXT

        def get_url_type(url: str, path: Optional[Path] = None) -> InputFormat:
            parsed_url = urllib.parse.urlparse(url)
            netloc = parsed_url.netloc
            input_type = None
            if path:
                input_type = get_text_file_type(path)
            if netloc in YOUTUBE_NETLOCS:
                return InputFormat.YOUTUBE_VIDEO
            # for this case we already know that the url was valid
            elif input_type and input_type == InputFormat.HTML:
                return InputFormat.WEB_PAGE
            else:
                raise ValueError(f"Unsupported url type: {url}")

        path = None
        try:
            path = cls._download_from_url(input_pointer)
            return get_url_type(input_pointer, path)
        except ValueError as e:
            logger.info(f"Failed to get url type: {e}, retrying with file type.")
        try:
            path = Path(input_pointer) if not path else path
            kind = filetype.guess(path)
            if kind:
                return InputFormat(kind.extension)
            else:
                return get_text_file_type(path)
        except (ValueError, UnicodeDecodeError) as e:
            logger.error(traceback.format_exc())
            extension = kind.extension if kind else path.suffix
            raise ValueError(f"Unsupported file type: {extension}.") from e

    @staticmethod
    def _download_from_url(url: str) -> Path:
        """Download the data from the url."""
        # get just the last part of the path without the query param
        final_path = urllib.parse.urlparse(url).path.split("/")[-1]
        tmp_path: Path = Path("/tmp") / str(uuid4()) / final_path
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537',
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error if the download fails
        with open(tmp_path, "wb") as f:
            f.write(response.content)
        return tmp_path

    @classmethod
    def _download_from_doc(cls, input_data: InputDocument) -> IngestedDocument:
        # get just the last part of the path without the query param
        path = cls._download_from_url(input_data.full_resource_url)
        file_type = cls._get_input_format(str(path.resolve()))
        document = IngestedDocument(
            data_pointer=path,
            input_format=file_type,
            **input_data.dict(),
        )
        return document


class S3ObjectIngestor(Ingestor):
    """
    Define the S3 ingestor.

    This class is used for ingesting data from S3.
    """

    @staticmethod
    def is_s3_url(url: str) -> bool:
        """Check if the url is an S3 url."""
        parsed_url = urllib.parse.urlparse(url)
        netloc = parsed_url.netloc
        if netloc == "s3.amazonaws.com":
            return True
        return False

    @classmethod
    def ingest_data(cls, input_data: InputDocument, bucket_name: str) -> IngestedDocument:
        """Ingest the data from S3."""
        # TODO: add s3 signature
        return cls._download_from_doc(input_data)


class WebPageIngestor(Ingestor):
    """
    Define the URL ingestor.

    This class is used for ingesting data from a URL.
    """

    @classmethod
    def ingest_data(cls, input_data: InputDocument, bucket_name: str) -> IngestedDocument:
        """Ingest the data from a URL."""
        doc = cls._download_from_doc(input_data)
        return doc


class RawUrlIngestor(Ingestor):
    """
    Define the raw URL ingestor.

    This class is used for ingesting data from a raw URL.
    """

    @classmethod
    def is_raw_url(cls, url: str) -> bool:
        """Check if the url is a raw url."""
        parsed_url = urllib.parse.urlparse(url)
        netloc = parsed_url.netloc
        file_type = cls._get_input_format(url)
        if netloc in YOUTUBE_NETLOCS or file_type == InputFormat.WEB_PAGE:
            return True
        return False

    @classmethod
    def ingest_data(cls, input_data: InputDocument, bucket_name: str) -> IngestedDocument:
        """Ingest the data from a raw URL."""
        url_type = cls._get_input_format(str(input_data.full_resource_url))
        if url_type == InputFormat.YOUTUBE_VIDEO:
            data_pointer = YoutubeLoader.extract_video_id(input_data.full_resource_url)
        else:
            data_pointer = input_data.full_resource_url
        document = IngestedDocument(
            data_pointer=data_pointer,
            input_format=url_type,
            **input_data.dict(),
        )
        return document

def ingest_strategy_factory(url: str) -> InputDataIngestStrategy:
    """Get the ingest strategy."""
    if S3ObjectIngestor.is_s3_url(url):
        return InputDataIngestStrategy.S3_FILE_DOWNLOAD
    elif RawUrlIngestor.is_raw_url(url):
        return InputDataIngestStrategy.RAW_URL
    else:
        return InputDataIngestStrategy.URL_DOWNLOAD


def ingestor_factory(document: InputDocument) -> Ingestor:
    mapping: dict[InputDataIngestStrategy, Ingestor] = {
        InputDataIngestStrategy.S3_FILE_DOWNLOAD: S3ObjectIngestor,
        InputDataIngestStrategy.URL_DOWNLOAD: WebPageIngestor,
        InputDataIngestStrategy.RAW_URL: RawUrlIngestor,
    }
    IngestorClass = mapping.get(document.input_data_ingest_strategy)
    if IngestorClass:
        return IngestorClass
    raise NotImplementedError(f"Unsupported input data ingest strategy: {document.input_data_ingest_strategy}") from e
