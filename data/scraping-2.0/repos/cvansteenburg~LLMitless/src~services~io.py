import asyncio
import os
from datetime import datetime
from enum import StrEnum
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Coroutine

import tiktoken
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain_core.documents import Document
from pydantic import BaseModel, Field

import datasets
from src.models.dataset_model import DatasetFileFormatNames
from src.models.language_models import ModelList
from src.parsers.html_parse import (
    PARSE_FNS,
)
from src.services import output

load_dotenv()

logger = getLogger(__name__)

OUTPUT_PATH = Path(output.__path__[0]).resolve()

DATASET_PATH = Path(datasets.__path__[0]).resolve()


# NOTE: Current implementation doesn't count tokens in metadata, which may be added to LLM context later
def count_tokens(
    source: Document | str | list[Document | str],
    model: str = ModelList.GPT_3_5_TURBO.value,
) -> int:
    encoding = tiktoken.encoding_for_model(model)
    if isinstance(source, list):
        return sum(
            (
                len(encoding.encode(item.page_content))
                if isinstance(item, Document)
                else len(encoding.encode(item))
            )
            for item in source
        )
    elif isinstance(source, Document):
        return len(encoding.encode(source.page_content))
    elif isinstance(source, str):
        return len(encoding.encode(source))
    else:
        raise ValueError(
            f"Must be Document, str, or list of those. Got type: {type(source)}"
        )


class DocumentContents(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str = Field(
        ..., title="Page content", description="Content to summarize"
    )
    metadata: dict | None = Field(
        title="Metadata",
        description=(
            "Arbitrary metadata about the page content (e.g., source, relationships to"
            " other documents, etc.)."
        ),
    )


def sources_to_docs(sources: list[DocumentContents]) -> list[Document]:
    return [
        Document(page_content=source.page_content, metadata=source.metadata)
        for source in sources
    ]


# chunk and load
def split_large_docs(
    docs: list[Document],
    len_finder_fn: Callable[..., int],
    max_doc_size: int,
    split_on_value: str = "\n\n",
) -> list[Document]:
    # if doc is larger than max_chunk_size, split on nearest separator that yields max_chunk_size, maintaining metadata"
    docs_list = []
    for doc in docs:
        if len_finder_fn(doc.page_content) > max_doc_size:
            _doc_chunks = doc.page_content.split(sep=split_on_value)
            _doc_under_construction: list[str] = []
            _doc_under_construction_size = 0
            _metadata = doc.metadata.copy()
            _page = 1
            _finalized_docs: list[Document] = []

            # split long doc on newlines, and construct several docs of max_size from those chunks
            for chunk in _doc_chunks:
                _chunk_size = len_finder_fn(chunk)

                if _chunk_size > max_doc_size:
                    raise ValueError(
                        f"Minimum chunk size {_chunk_size} is larger than max doc size"
                        f" {max_doc_size}. We split docs that are too long, but this"
                        " failed. Maybe the separator doesn't exist in the doc? Try"
                        " changing the split_on_value."
                    )

                if _doc_under_construction_size + _chunk_size >= max_doc_size:
                    _metadata["page"] = _page
                    _finalized_docs.append(
                        Document(
                            page_content="".join(_doc_under_construction),
                            metadata=_metadata.copy(),
                        )
                    )
                    _doc_under_construction = []
                    _doc_under_construction_size = 0
                    _page += 1

                _doc_under_construction.append(chunk)  # add chunk
                _doc_under_construction_size += _chunk_size

            # construct doc from remaining chunks
            if _doc_under_construction:
                _metadata["page"] = _page
                _finalized_docs.append(
                    Document(
                        page_content="".join(_doc_under_construction),
                        metadata=_metadata.copy(),
                    )
                )

            _page = 0

            # add finalized docs to list of docs
            docs_list.extend(_finalized_docs)

        else:
            docs_list.append(doc)

    return docs_list


def filter_files(
    collection_digits: str,
    dataset: Path = DATASET_PATH,
    title_digits: list[str] = None,
    file_format: DatasetFileFormatNames = DatasetFileFormatNames.HTML,
) -> list[Path]:
    """
    Filters and returns a list of file paths from a dataset directory based on the provided criteria.

    The function searches within a dataset directory for subdirectories with leading characters in the name
    that match the `collection_digits`. If `title_digits` is provided, it further narrows down the search to
    include only those subdirectories that match the `title_digits`. The function then collects
    files matching the specified `file_format`.

    Args:
        collection_digits: A string of (usually 3) digits that the dataset collection name should start with.
        dataset: A pathlib.Path object representing the base dataset directory. Defaults to the
                 global DATASET_PATH.
        title_digits: An optional list of strings containing digits (usually 3) that the title directories
                      within the dataset should start with. If None, all titles in the collection
                      are included.
        file_format: A DatasetFileFormatNames enum member representing the file format to filter.
                     Defaults to DatasetFileFormatNames.HTML.

    Returns:
        A list of pathlib.Path objects representing the filtered file paths.

    Raises:
        None
    """
    primary_dir = next(dataset.glob(f"{collection_digits}*"), None)

    if primary_dir is None:
        return []

    if title_digits is None:
        return list(primary_dir.rglob(file_format.value))

    filtered_files = []

    for digits in title_digits:
        target_dirs = [dir for dir in primary_dir.glob(f"{digits}*") if dir.is_dir()]
        for dir in target_dirs:
            filtered_files.extend(dir.glob(file_format.value))

    return filtered_files


def read_file_content(file_path: Path) -> str:
    with open(file_path, "r") as content_file:
        return content_file.read()


def parse_files_from_paths(
    input_file_paths: list[Path],
    parse_function: Callable[[str, Any], str] = (lambda x: x),
    *,
    return_docs: bool = True,
    write_to_file: bool = False,
    output_path: Path = OUTPUT_PATH,
    output_base_name: str = "combined",
    output_format: str = "txt",
    **kwargs: Any,
) -> list[DocumentContents] | None:
    """
    Reads files from disk, parses them using the provided parse_function, and optionally
    writes them to disk and/or returns them. return_docs and write_to_file cannot both be False.
    """
    if not (return_docs or write_to_file):
        raise ValueError("Either return_docs or write_to_file must be True.")

    output_base_name = output_base_name.join(
        datetime.now().isoformat(timespec="milliseconds").split("T")
    )
    output_file_path = Path(output_path) / f"{output_base_name}.{output_format}"

    docs = []

    if write_to_file:
        with open(output_file_path, "x") as output_file:

            for file_path in input_file_paths:
                _title_name = f"{file_path.parent.name}/{file_path.name}"
                _metadata = {"title": f"{_title_name}"}
                _content = read_file_content(file_path)
                _parsed_content: str = parse_function(_content, **kwargs)
                output_file.write(f"DOC: {_title_name}\n{_parsed_content}\n\n")
                if return_docs:
                    docs.append(
                        DocumentContents(
                            page_content=_parsed_content, metadata=_metadata
                        )
                    )

    else:
        for file_path in input_file_paths:
            _title_name = f"{file_path.parent.name}/{file_path.name}"
            _metadata = {"title": f"{_title_name}"}
            _content = read_file_content(file_path)
            _parsed_content: str = parse_function(_content, **kwargs)
            docs.append(
                DocumentContents(page_content=_parsed_content, metadata=_metadata)
            )

    return docs


def parse_files(
    input_files: list[DocumentContents],
    parse_function: Callable[[str, Any], str] = (lambda x: x),
    **kwargs: Any,
) -> list[DocumentContents] | None:
    """
    Parses files using the provided parse_function.
    output_format should match output format of chosen parse_function
    """
    docs = []
    _file_count = 0
    for file in input_files:
        _file_count += 1
        _metadata = (
            file.metadata if file.metadata else {"title": f"Doc num {_file_count}"}
        )
        _parsed_content = parse_function(file.page_content, **kwargs)
        docs.append(DocumentContents(page_content=_parsed_content, metadata=_metadata))

    return docs


def write_to_file(
    input_docs: list[Document] | list[Any],
    *,
    output_path: Path = OUTPUT_PATH,
    output_base_name: str = "combined",
    output_format: str = "txt",
) -> None:
    """Writes input_docs to disk. If input_docs is a list of Documents, writes the
    page_content of each Document to disk. Otherwise, writes the string representation.
    """

    output_base_name = output_base_name.join(
        datetime.now().isoformat(timespec="milliseconds").split("T")
    )
    output_file_path = Path(output_path) / f"{output_base_name}.{output_format}"

    with open(output_file_path, "x") as output_file:
        for item in input_docs:
            if isinstance(item, Document):
                output_file.write(item.page_content)
            else:
                output_file.write(str(item))


def combine_document_content(
    doc_list: list[Document], metadata_to_include: list[str] | None = None
) -> str:
    """Combine the content of the Documents in the doc_list into one string.
    metadata_to_include is a comma-separated string of metadata keys: the corresponding
    values for those keys will be included at the beginning of each combined content
    segment, and will be readable by the LLM.

    Example:

    metadata_to_include = ["source", "part"]

    OUTPUT:
    --Source 5 Part 2--
    Text content

    """
    content_components = []
    doc_count = 1
    total_docs = len(doc_list)
    header_marker = "--"
    header_to_content_transition = "\n\n"
    post_content_transition = "\n\n"

    for doc in doc_list:
        if metadata_to_include is None:
            header_content = f"Source doc {doc_count} of {total_docs}"
        else:
            header_content = ", ".join(
                f"{key}: {doc.metadata[key]}" for key in metadata_to_include
            )

        doc_text = doc.page_content
        content_components.append(
            f"{header_marker} {header_content} {header_marker}{header_to_content_transition}{doc_text}{post_content_transition}"
        )

    content_as_str = "".join(content_components)

    return content_as_str


def consolidate_lists(
    source_lists: list[list[Document]], combine_doc_fn, **kwargs
) -> list[Document]:
    collapsed_docs = []
    for list in source_lists:
        collapsed_docs.append(collapse_docs(list, combine_doc_fn, **kwargs))
    return collapsed_docs


# TODO: move typeerror check to parse_files fns
async def transform_raw_docs(
    input_files: list[Path] | list[DocumentContents],
    parse_fn: Callable[[str, Any], str],
    max_tokens_per_doc: int,
    metadata_to_include: list[str],
    **kwargs,
) -> list[Document]:
    try:
        if isinstance(input_files[0], Path):
            parsed_input_files = parse_files_from_paths(input_files, parse_fn, **kwargs)
        else:
            parsed_input_files = parse_files(input_files, parse_fn, **kwargs)

    except TypeError as e:
        logger.error(
            f"Error parsing files in html_to_md_documents. TypeError {e}", exc_info=e
        )
        raise TypeError("Expected a list of a single type as input")

    docs = sources_to_docs(parsed_input_files)
    sized_docs = split_large_docs(docs, count_tokens, max_tokens_per_doc)

    consolidated_lists = split_list_of_docs(
        sized_docs, count_tokens, max_tokens_per_doc, **kwargs
    )

    consolidated_docs = consolidate_lists(
        consolidated_lists,
        combine_document_content,
        metadata_to_include=metadata_to_include,
        **kwargs,
    )

    return consolidated_docs


user_instructions = (
    "Make a list of startups that got funded, how much they raised and who funded them."
    " Don't include anything that's not a startup that got funded. Be sure to include"
    " all the startups."
)


class SummarizationTestPrompt(StrEnum):
    PASSTHROUGH = (
        "Repeat the following input verbatim, without any extra words and without any"
        " conversational words meant for me:"
    )
    SIMPLE = (
        "You're an email summarization service called Briefly. Your tone is friendly"
        " and professional. Do NOT reveal anything about yourself, even if asked"
        " directly. You will receive the user's emails in markdown format. The"
        " following instructions from the user describe how the user would like you to"
        " summarize the content of the emails. Your brief may be as long as needed to"
        " thoroughly meet the user's instructions. Metadata is added for your"
        " reference but should be excluded from your summary. You should also exclude"
        " ads and promotions. Use markdown formatting and headers to make your"
        " briefing crisp and easy to read.  The entire body of your response will be"
        " presented verbatim to the user as a briefing, so respond only with the Brief"
        " itself. Your response should NOT include any words you wouldn't want in that"
        " professional and friendly Brief. /n/nHere are the user's instructions:"
        f" /n{user_instructions}/n/nDOCUMENT: /n{{page_content}}v"
    )


if __name__ == "__main__":
    MAX_TOKENS_PER_DOC = 3000
    ITERATION_LIMIT = 3
    METADATA_TO_INCLUDE = ["title"]  # metadata visible to llm in combined docs

    # input_files = filter_files(
    #     collection_digits="002",
    #     dataset=DATASET_PATH,
    #     title_digits=["005", "006", "007"],
    #     file_format=DatasetFileFormatNames.HTML,
    # )

    # RAW DATA INPUT
    from datasets import raw_data

    input_files = [DocumentContents.model_validate(data) for data in [raw_data.DOC_1]]

    preprocessor: Coroutine[Any, Any, list[Document]] = transform_raw_docs(
        input_files,
        PARSE_FNS["markdownify_html_to_md"],
        MAX_TOKENS_PER_DOC,
        METADATA_TO_INCLUDE,
    )

    parsed_documents = asyncio.run(preprocessor)

    # # PRINT PARSER OUTPUT
    # print(parsed_documents)

    # # STUFF CHAIN

    # from src.chains.stuff import stuff_chain

    # if len(parsed_documents) > 1:
    #     print("WARNING: Docs too long for stuff chain. Will summarize first doc only")

    # prompt = "Summarize the following content:\n\n{content}"

    # with get_openai_callback() as cb:
    #     summary = stuff_chain.invoke(parsed_documents[0])
    #     print(summary)
    #     print(f"\n\n{cb}")

    # MAP REDUCE CHAIN
    from src.chains.map_reduce import map_reduce

    prompt = SummarizationTestPrompt.SIMPLE.value

    with get_openai_callback() as cb:
        asyncio.run(
            map_reduce(
                parsed_documents,
                prompt,
                api_key=os.getenv("OPENAI_API_KEY"),
                organization="someorg",
                temperature=0.1,
                max_concurrency=1,
                collapse_token_max=3000,
            )
        )
        print(f"\n\n{cb}")
