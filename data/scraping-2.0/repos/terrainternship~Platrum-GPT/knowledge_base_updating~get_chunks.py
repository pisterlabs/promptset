import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, MarkdownTextSplitter


def split_to_chunks_md_text(markdown_text, c_size, c_overlap):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    fragments = [fragment.strip() for fragment in re.split(r"<[^>]+>|[\ufeff]", markdown_text) if fragment.strip()]
    source_chunks = []
    for fragment in fragments:
        source_chunks.extend(markdown_splitter.split_text(fragment))

    text_splitter = MarkdownTextSplitter(
        chunk_size=int(c_size), chunk_overlap=int(c_overlap)
    )
    return text_splitter.split_documents(source_chunks)
