import re
import numpy as np

from typing import List
from pathlib import Path

from langchain.schema import Document
from langchain.document_loaders import PyPDFium2Loader

from onepoint_document_chat.log_init import logger
from onepoint_document_chat.config import cfg


FILE_NAME = "file_name"
PAGE = "page"


def extract_meta_data(path: Document, i: int):
    path.metadata[FILE_NAME] = re.sub(r".+[\\/]", "", path.metadata["source"])
    path.metadata[PAGE] = [i + 1]


def load_pdfs(path: Path) -> List[Document]:
    """
    Loads the PDFs and extracts a document per page.
    The page details are added to the extracted metadata

    Parameters:
    path (Path): The path where the PDFs are saved.

    Returns:
    List[Document]: Returns a list of values
    """
    assert path.exists()
    all_pages = []
    for pdf in path.glob("*.pdf"):
        pages = extract_single_pdf(pdf)
        all_pages.extend(pages)
        logger.info(f"Processed {pdf}, all_pages size: {len(all_pages)}")
    log_stats(all_pages)
    return all_pages


def merge_docs(doc1: Document, doc2: Document) -> Document:
    new_page_content = f"{doc1.page_content}\n\n{doc2.page_content}"
    new_metadata = {
        FILE_NAME: f"{doc1.metadata[FILE_NAME]}, {doc2.metadata[FILE_NAME]}",
        PAGE: doc1.metadata[PAGE] + doc2.metadata[PAGE],
    }
    return Document(page_content=new_page_content, metadata=new_metadata)


def combine_documents(documents: List[Document]) -> List[Document]:
    new_docs: List[Document] = []
    cur_doc = None
    merge_limit_chars = cfg.doc_min_length

    for d in documents:
        if len(d.page_content) > merge_limit_chars:
            if cur_doc is None:
                new_docs.append(
                    Document(page_content=d.page_content, metadata=d.metadata)
                )
            else:
                new_docs.append(merge_docs(cur_doc, d))
                cur_doc = None
        else:
            if cur_doc is None:
                cur_doc = Document(page_content=d.page_content, metadata=d.metadata)
            else:
                cur_doc = merge_docs(cur_doc, d)
                if len(cur_doc.page_content) > merge_limit_chars:
                    new_docs.append(cur_doc)
                    cur_doc = None
    if cur_doc is not None:
        new_docs.append(cur_doc)
    return new_docs


def extract_single_pdf(pdf: Path) -> List[Document]:
    loader = PyPDFium2Loader(str(pdf.absolute()))
    pages: List[Document] = loader.load_and_split()
    for i, p in enumerate(pages):
        extract_meta_data(p, i)
    return pages


def log_stats(documents: List[Document]):
    logger.info(f"Total number of documents {len(documents)}")
    counts = []
    for d in documents:
        counts.append(count_words(d))
    logger.info(f"Tokens Max {np.max(counts)}")
    logger.info(f"Tokens Min {np.min(counts)}")
    logger.info(f"Tokens Min {np.mean(counts)}")


def count_words(document: Document) -> int:
    splits = [s for s in re.split("[\s,.!?]", document.page_content) if len(s) > 0]
    return len(splits)


def write_to_temp_folder(documents: List[Document]) -> Path:
    target_folder = cfg.extraction_text_folder
    for doc in documents:
        file_name = re.sub(r"(.+)\..+", r"\1", doc.metadata[FILE_NAME])
        page = doc.metadata[PAGE]
        target_path = target_folder / f"{file_name}_{page}.txt"
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)
            logger.info(f"Wrote to {target_path}")
    return target_folder


if __name__ == "__main__":
    from onepoint_document_chat.log_init import logger
    from onepoint_document_chat.config import cfg
    from onepoint_document_chat.service.text_enhancement import enhance_text

    documents: List[Document] = load_pdfs(cfg.data_folder)
    documents = combine_documents(documents)
    target_folder = write_to_temp_folder(documents)
    # for file in target_folder.glob("*.txt"):
    #     enhance_text(file)
