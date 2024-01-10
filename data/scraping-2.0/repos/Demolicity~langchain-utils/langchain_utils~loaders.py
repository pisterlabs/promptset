#!/usr/bin/env python3

from typing import TYPE_CHECKING, Literal

from langchain_utils.utils import extract_github_info, get_github_file_raw_url
from langchain_utils.config import (
    TESSERACT_OCR_DEFAULT_LANG,
)

if TYPE_CHECKING:
    from langchain.docstore.document import Document


def load_youtube_url(youtube_url: str) -> list['Document']:
    from langchain.document_loaders import YoutubeLoader

    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
    docs = loader.load()
    return docs


def load_pdf(
    pdf_path: str,
    use_ocr_if_no_text_detected_on_page: bool = False,
    ocr_language: str = TESSERACT_OCR_DEFAULT_LANG,
) -> list['Document']:
    if use_ocr_if_no_text_detected_on_page:
        from langchain_utils.document_loaders import PyMuPDFLoaderWithFallbackOCR

        loader_cls = PyMuPDFLoaderWithFallbackOCR
        load_kwargs = {'ocr_language': ocr_language}
    else:
        from langchain.document_loaders import PyMuPDFLoader

        loader_cls = PyMuPDFLoader
        load_kwargs = {}

    loader = loader_cls(pdf_path)
    docs = loader.load(**load_kwargs)
    return docs


def load_url(urls: list[str], javascript: bool = False) -> list['Document']:
    from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader

    if javascript:
        loader_class = SeleniumURLLoader
        kwargs = {}
    else:
        loader_class = UnstructuredURLLoader
        # headers = {
        #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
        # }

        # kwargs = {'headers': headers}
        # You are using old version of unstructured. The headers parameter is ignored
        kwargs = {}

    from unstructured.partition.html import partition_html

    partition_html(url='https://mp.weixin.qq.com/s/FsrDnCFKGD-FzP5YD76tbA')
    loader = loader_class(urls=urls, **kwargs)
    docs = loader.load()

    return docs


def load_text(path: str, encoding: str | None = None) -> list['Document']:
    from langchain.document_loaders import TextLoader

    loader = TextLoader(path, encoding=encoding)
    docs = loader.load()
    return docs


def load_html(
    path: str, open_encoding: str | None = None, bs_kwargs: dict | None = None
) -> list['Document']:
    from langchain.document_loaders import BSHTMLLoader

    loader = BSHTMLLoader(path, open_encoding=open_encoding, bs_kwargs=bs_kwargs)
    docs = loader.load()
    return docs


UnstructuredLoadingMode = Literal["single", "elements"]


def load_word(path: str, mode: UnstructuredLoadingMode = "single") -> list['Document']:
    # UnstructuredWordDocumentLoader
    from langchain.document_loaders import UnstructuredWordDocumentLoader

    loader = UnstructuredWordDocumentLoader(path, mode=mode)
    docs = loader.load()
    return docs


def load_github_raw(
    github_url: str, github_revision: str = 'master', github_path: str = 'README.md'
) -> list['Document']:
    from langchain.requests import TextRequestsWrapper
    from langchain.docstore.document import Document

    github_info = extract_github_info(github_url)
    if github_info is None:
        raise ValueError(f'Invalid GitHub URL: {github_url}')
    github_info |= {'revision': github_revision, 'file_path': github_path}
    url = get_github_file_raw_url(**github_info)
    text = TextRequestsWrapper().get(url)

    docs = [Document(page_content=text, metadata={'url': url})]
    return docs
