from langchain.schema import Document
from langchain.document_loaders import PyPDFium2Loader

from typing import List, Optional
from pathlib import Path

from hrranker.log_init import logger
from hrranker.config import cfg


def extract_data(path: Path, filter: Optional[str] = None) -> List[Document]:
    assert path.exists(), f"Path {path} does not exist."
    res: List[Document] = []
    pdfs = list(path.glob("*.pdf"))
    logger.info(f"There are {len(pdfs)} physical documents.")
    for pdf in pdfs:
        if filter is None or filter in pdf.stem:
            new_document = convert_pdf_to_document(pdf)
            res.append(new_document)
    return res


def convert_pdf_to_document(pdf: Path) -> Document:
    loader = PyPDFium2Loader(str(pdf.absolute()))
    pages: List[Document] = loader.load()
    metadata = pages[0].metadata
    pdf_content = ""
    for p in pages:
        pdf_content += p.page_content
    new_document = Document(page_content=pdf_content, metadata=metadata)
    return new_document


if __name__ == "__main__":
    path = cfg.doc_location
    docs = documents = extract_data(path)
    logger.info(f"There are {len(documents)} documents")
    logger.info(f"{docs[0].page_content[:100]}")
