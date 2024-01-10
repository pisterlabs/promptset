"""Test cases for the split_markdown module."""
from langchain.schema import Document

from models.split_markdown import RecursiveMarkdownTextSplitter


def test_recursive_markdown_text_splitter() -> None:
    """It splits markdown correctly."""
    page_content = """
    Introduction
    ## Section One
    This is the first line of section one.
    This is the second line of section one.
    This is the third line of section one.
    This is the fourth line of section one.
    ## Section Two
    This is the first line of section two.
    ## Section Three
    This is the first line of section three.
    """
    splitter = RecursiveMarkdownTextSplitter(
        headers_to_split_on=[("##", "Header 2")],
        title_header_separator="|",
        chunk_size=100,
        chunk_overlap=0,
    )
    doc = Document(
        metadata={"title": "Title"},
        page_content=page_content,
    )
    results = splitter.split_documents([doc])
    assert len(results) == 5
    assert results[0].page_content == "Introduction"
    assert results[0].metadata["title"] == "Title"
    assert results[1].page_content == "This is the first line of section one.\nThis is the second line of section one."
    assert results[1].metadata["title"] == "Title|Section One"
    assert results[2].page_content == "This is the third line of section one.\nThis is the fourth line of section one."
    assert results[2].metadata["title"] == "Title|Section One"
    assert results[3].page_content == "This is the first line of section two."
    assert results[3].metadata["title"] == "Title|Section Two"
    assert results[4].page_content == "This is the first line of section three."
    assert results[4].metadata["title"] == "Title|Section Three"
