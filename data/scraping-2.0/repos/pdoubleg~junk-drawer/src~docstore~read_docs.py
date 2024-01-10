import os
from pathlib import Path
from typing import Union, List

import re
import pypdf
import fitz
from html2text import html2text
from markdownify import markdownify
from llama_index import Document
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.node_parser import SimpleNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


def remove_multiple_newlines(page_md):
    page_md = re.sub(r"\n\s*\n", "\n\n", page_md)
    return page_md


def get_file_name(path):
    file_name = os.path.splitext(os.path.basename(path))[0]
    return file_name


def create_relationships(nodes):
    for i in range(len(nodes)):
        if i > 0:
            node = nodes[i]
            previous_node = nodes[i - 1]

            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=previous_node.id_
            )
            previous_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node.id_
            )

    return nodes


def parse_pdf_fitz(path: Path, docname: str, chunk_chars: int, overlap: int):
    file = fitz.open(path)
    file_name = get_file_name(path)
    split = ""
    page_docs: List[str] = []
    pages: List[str] = []
    nodes: List[TextNode] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        page_doc = page.get_text("text", sort=False)
        page_docs.append(page_doc)
        split += page.get_text("text", sort=False)
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            nodes.append(
                TextNode(
                    text=split[:chunk_chars],
                    metadata={"document": f"{docname}", 
                              "pages": f"{pg}",
                              "file_name": f"{file_name}"},
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        nodes.append(
            TextNode(
                text=split[:chunk_chars],
                metadata={"document": f"{docname}", 
                          "pages": f"{pg}",
                          "file_name": f"{file_name}"},
            )
        )
    nodes = create_relationships(nodes)

    all_pages = "".join(page_docs)
    docs = Document(
        text=all_pages,
        id_=f"{file_name}",
        metadata={"document": f"{docname}"},
    )

    return nodes, docs


def parse_pdf(path: Union[Path, List[Path]], docname: str, chunk_chars: int, overlap: int):
    pdfFileObj = open(path, "rb")
    file_name = get_file_name(path)
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    page_docs: List[str] = []
    pages: List[str] = []
    nodes: List[TextNode] = []
    for i, page in enumerate(pdfReader.pages):
        page_doc = page.extract_text()
        page_docs.append(page_doc)
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            nodes.append(
                TextNode(
                    text=split[:chunk_chars],
                    metadata={"document": f"{docname}", 
                              "pages": f"{pg}",
                              "file_name": f"{file_name}"},
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        nodes.append(
            TextNode(
                text=split[:chunk_chars],
                metadata={"document": f"{docname}", 
                          "pages": f"{pg}",
                          "file_name": f"{file_name}"},
            )
        )

    all_pages = "".join(page_docs)
    docs = Document(
        text=all_pages,
        id_=f"{file_name}",
        metadata={"document": f"{docname}"},
    )
    pdfFileObj.close()

    return nodes, docs


def parse_html(
    path: Union[Path, List[Path]], docname: str, chunk_chars: int, overlap: int):
    paths = []

    if isinstance(path, list):
        paths = [Path(p) if isinstance(p, str) else p for p in path]
    elif isinstance(path, str):
        path = Path(path)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths = list(path.glob("*.htm*"))

    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap,
        length_function=len,
    )
    parser = SimpleNodeParser(text_splitter=text_splitter)

    for file_path in paths:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
            file_name = get_file_name(file_path)
            text = html2text(text)
            text = markdownify(text, heading_style="ATX")
            text = remove_multiple_newlines(text)

            doc = Document(
                text=text,
                id_=f"{file_name}",
                metadata={"filename": f"{docname}", 
                          "file_path": f"{file_path}"},
            )
            docs.append(doc)

    nodes = parser.get_nodes_from_documents(docs)

    return nodes, docs


def read_docs(
    path: str,
    docname: str,
    chunk_chars: int = 1024,
    overlap: int = 0,
    force_pypdf: bool = False,
):
    """Parse a document into chunks."""
    path_obj = Path(path)

    if path_obj.is_file():
        path_str_list = [str(path_obj)]
    elif path_obj.is_dir():
        path_str_list = [
            str(file_path) for file_path in path_obj.iterdir() if file_path.is_file()
        ]
    else:
        raise ValueError("Invalid path provided.")

    nodes = []
    docs = []
    for path_str in path_str_list:
        file_extensions = path_str.split(".")[-1]

        if file_extensions == "pdf":
            if force_pypdf:
                node, doc = parse_pdf(path_str, docname, chunk_chars, overlap)
            else:
                try:
                    node, doc = parse_pdf_fitz(path_str, docname, chunk_chars, overlap)
                except ImportError:
                    node, doc = parse_pdf(path_str, docname, chunk_chars, overlap)
            nodes.append(node)
            docs.append(doc)
        elif file_extensions in ["html", "htm"]:
            node, doc = parse_html(path_str_list, docname, chunk_chars, overlap)
            nodes.append(node)
            docs.append(doc)

    return nodes[0], docs

