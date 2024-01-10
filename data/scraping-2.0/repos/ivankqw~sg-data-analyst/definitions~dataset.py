from __future__ import annotations
from dataclasses import dataclass, asdict
from langchain.schema import Document
import json


@dataclass
class Dataset:
    """Dataset class to store dataset information"""
    id: str
    name: str
    description: str

    def to_document(self) -> Document:
        """Converts the Dataset to a Document by serializing it to a JSON string.
        
        :return: the document :class `Document`
        """
        return Document(page_content=json.dumps(asdict(self)))

    @staticmethod
    def from_document(doc: Document) -> Dataset:
        """Converts a Document to a Dataset by deserializing it from a JSON string.
        
        :param doc: the document :class `Document`

        :return: the dataset :class `Dataset`
        """
        data = json.loads(doc.page_content)
        return Dataset(**data)
