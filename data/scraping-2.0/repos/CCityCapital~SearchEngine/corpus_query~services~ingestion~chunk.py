import argparse
from enum import Enum
import logging
from typing import Generator, Optional

from langchain.text_splitter import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from pydantic import BaseModel


def chunk_file_by_line(corpus_string: str) -> Generator[str, None, None]:
    """
    Chunk a corpus string into smaller strings.
    """
    prev_line = None
    for line in corpus_string.split("\n"):
        stripped_line = line.strip()
        if len(stripped_line) == 0:
            continue
        if prev_line is not None:
            yield " ".join([prev_line, stripped_line])

        prev_line = stripped_line


class FileTypes(Enum):
    MARKDOWN = "md"
    TEXT = "txt"
    HTML = "html"
    PDF = "pdf"


class ChunkingOptions(BaseModel):
    chunk_size: int
    chunk_overlap: int


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def chunk_by_file_type(
    corpus_text: str, file_type: FileTypes, options: Optional[ChunkingOptions] = None
) -> list[Document]:
    logging.info("chunking value: '%s'", corpus_text)
    if options is None:
        options = ChunkingOptions(chunk_size=100, chunk_overlap=20)

    if file_type == FileTypes.MARKDOWN:
        txt_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        markdown_docs = txt_splitter.split_text(corpus_text)

        txt_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
        )
        return txt_splitter.split_documents(markdown_docs)

    elif file_type == FileTypes.HTML:
        html_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
        )

        return [Document(s) for s in html_splitter.split_text(corpus_text)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
    )
    return [Document(page_content=s) for s in text_splitter.split_text(corpus_text)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file", type=str, help="Path to corpus file.")
    args = parser.parse_args()

    with open(args.corpus_file, "r", encoding="utf-8") as f:
        for d in chunk_by_file_type(f.read(), FileTypes.MARKDOWN):
            print(d)
