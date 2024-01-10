from repolya._log import logger_rag

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter

import re


##### RecursiveCharacterTextSplitter
def get_RecursiveCharacterTextSplitter(text_chunk_size=3000, text_chunk_overlap=300):
    ##### default list is ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n\n\n\n", "\n\n\n\n", "\n\n\n", "\n\n", "\n", " ", ""],
        chunk_size = text_chunk_size,
        chunk_overlap = text_chunk_overlap,
        length_function = len,
        add_start_index = True,
        is_separator_regex = False,
    )
    return text_splitter


##### MarkdownHeaderTextSplitter
def get_MarkdownHeaderTextSplitter():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return md_splitter


##### split general
def split_docs_recursive(_docs, text_chunk_size, text_chunk_overlap):
    text_splitter = get_RecursiveCharacterTextSplitter(text_chunk_size, text_chunk_overlap)
    splited_docs = text_splitter.split_documents(_docs)
    _clean_splited_docs = []
    for doc in splited_docs:
        if doc.page_content.strip() != '':
            _clean_splited_docs.append(doc)
    logger_rag.info(f"split {len(_docs)} docs to {len(_clean_splited_docs)} splited_docs")
    return _clean_splited_docs


##### split pdf
def split_pdf_docs_recursive(_docs, text_chunk_size, text_chunk_overlap):
    text_splitter = get_RecursiveCharacterTextSplitter(text_chunk_size, text_chunk_overlap)
    splited_docs = text_splitter.split_documents(_docs)
    for i in splited_docs:
        _m = i.metadata
        _pdf = _m['file_path'].split('/')[-1]
        _page = _m['page']
        i.metadata['source'] = f"{_pdf}, p{_page}"
    logger_rag.info(f"split {len(_docs)} docs to {len(splited_docs)} splited_docs")
    return splited_docs


def split_pdf_text_recursive(_text, _fp, text_chunk_size, text_chunk_overlap):
    text_splitter = get_RecursiveCharacterTextSplitter(text_chunk_size, text_chunk_overlap)
    _docs = text_splitter.create_documents([_text])
    _n = 0
    for i in _docs:
        _m = i.metadata
        _m['file_path'] = _fp
        _pdf = _m['file_path'].split('/')[-1]
        i.metadata['source'] = f"{_pdf}, s{_n}"
        _n += 1
    logger_rag.info(f"split text to {len(_docs)} docs")
    return _docs
