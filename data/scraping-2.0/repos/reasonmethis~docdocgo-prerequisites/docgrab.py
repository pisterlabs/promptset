from typing import Iterable

import os
import json
from dotenv import load_dotenv

from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document

load_dotenv()


class JSONLDocumentLoader:
    def __init__(self, file_path: str, max_docs=None) -> None:
        self.file_path = file_path
        self.max_docs = max_docs

    def load(self) -> list[Document]:
        docs = load_docs_from_jsonl(self.file_path)
        if self.max_docs is None or self.max_docs > len(docs):
            return docs
        return docs[: self.max_docs]


def load_confluence() -> list[Document]:
    loader = ConfluenceLoader(
        url=os.getenv("ATLASSIAN_URL"),
        username=os.getenv("ATLASSIAN_USERNAME"),
        api_key=os.getenv("ATLASSIAN_API_KEY"),
    )
    documents = loader.load(
        space_key=os.getenv("CONFLUENCE_SPACE"), include_attachments=False, limit=50
    )
    return documents


def save_docs_to_jsonl(docs: Iterable[Document], file_path: str, mode="a") -> None:
    with open(file_path, mode) as f:
        for doc in docs:
            f.write(doc.json() + "\n")


def load_docs_from_jsonl(file_path: str) -> list[Document]:
    docs = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            doc = Document(**data)
            docs.append(doc)
    return docs
