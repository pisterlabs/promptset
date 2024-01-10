from io import BytesIO
import fitz
from langchain.docstore.document import Document
from typing import List, Any, Optional
from abc import abstractmethod, ABC
from copy import deepcopy
from hashlib import md5
import docx2txt
from utils import img2txt, strip_consecutive_newlines

class File(ABC):
    def __init__(
            self,
            name: str,
            id: str,
            metadata: Optional[dict[str, Any]] = None,
            docs: Optional[List[Document]] = None,
    ):
        self.name = name
        self.id = id
        self.metadata = metadata or {}
        self.docs = docs or []

    @classmethod
    @abstractmethod
    def from_bytes(cls, file: BytesIO) -> "File":
        """Creates a File from a BytesIO object"""
    def __repr__(self) -> str:
        return (
            f"File(name={self.name}, id={self.id},"
            f" metadata={self.metadata}"
        )
    
    def __str__(self) -> str:
        return f"File(name={self.name}, id={self.id}, metadata={self.metadata})"
    
    def copy(self) -> "File":
        """Create a deep copy of this File"""
        return self.__class__(
            name=self.name,
            id=self.id,
            metadata=deepcopy(self.metadata),
            docs=deepcopy(self.docs),
        )

class docxFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "docxFile":
        text = docx2txt.process(file)
        text = strip_consecutive_newlines(text)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read().hexdigest()), docs=[doc])


class pdfFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "pdfFile":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        docs= []
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            text = strip_consecutive_newlines(text)
            doc = Document(page_content=text.strip())
            doc.metadata["page"] = i + 1
            doc.metadata["source"] = f"p-{i+1}"
            docs.append(doc)
        file.seek(0)
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=docs)
    

class txtFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "txtFile":
        text = file.read().decode("utf-8")
        text = strip_consecutive_newlines(text)
        file.seek(0)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])


class imgFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "imgFile":
        text = img2txt(file)
        text = strip_consecutive_newlines(text)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])

def read_file(file: BytesIO) -> File:
    """Reads an uploaded file and returns a File object"""
    if file.name.lower().endswith(".docx"):
        return docxFile.from_bytes(file)
    elif file.name.lower().endswith(".pdf"):
        return pdfFile.from_bytes(file)
    elif file.name.lower().endswith(".txt"):
        return txtFile.from_bytes(file)
    elif file.name.lower().endswith(".png") or file.name.lower().endswith(".jpeg") \
        or file.name.lower().endswith(".tiff") or file.name.lower().endswith(".jpg"):
        return imgFile.from_bytes(file)
    else:
        raise NotImplementedError(f"File type {file.name.split('.')[-1]} not supported.")