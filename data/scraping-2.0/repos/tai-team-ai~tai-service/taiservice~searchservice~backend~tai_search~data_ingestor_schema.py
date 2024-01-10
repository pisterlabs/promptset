"""Define the data ingestor schemas."""
from enum import Enum
from typing import Optional
from pydantic import Field
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter
from ..shared_schemas import (
    BaseClassResourceDocument,
    StatefulClassResourceDocument,
)


class InputFormat(str, Enum):
    """Define the supported input formats."""

    PDF = "pdf"
    GENERIC_TEXT = "generic_text"
    LATEX = "latex"
    MARKDOWN = "markdown"
    HTML = "html"
    WEB_PAGE = "web_page"
    RAW_URL = "raw_url"
    YOUTUBE_VIDEO = "youtube_video"


class MarkdownExtension(str, Enum):
    """Define the markdown extensions."""

    MARKDOWN = ".markdown"
    MD = ".md"
    MKD = ".mkd"
    MDWN = ".mdwn"
    MDOWN = ".mdown"
    MDTXT = ".mdtxt"
    MDTEXT = ".mdtext"
    TXT = ".text"


class LatexExtension(str, Enum):
    """Define the latex extensions."""

    TEX = ".tex"
    LATEX = ".latex"


class InputDataIngestStrategy(str, Enum):
    """Define the input types."""

    S3_FILE_DOWNLOAD = "s3_file_download"
    URL_DOWNLOAD = "url_download"
    RAW_URL = "raw_url"
    # WEB_CRAWL = "web_crawl"



class InputDocument(BaseClassResourceDocument):
    """Define the input document."""

    input_data_ingest_strategy: InputDataIngestStrategy = Field(
        ...,
        description="The strategy for ingesting the input data.",
    )


class IngestedDocument(StatefulClassResourceDocument):
    """Define the ingested document."""

    input_format: InputFormat = Field(
        ...,
        description="The format of the input document.",
    )
    loader: Optional[BaseLoader] = Field(
        default=None,
        description="The loader for the ingested document.",
    )
    splitter: Optional[TextSplitter] = Field(
        default=None,
        description="The splitter for the ingested document.",
    )
