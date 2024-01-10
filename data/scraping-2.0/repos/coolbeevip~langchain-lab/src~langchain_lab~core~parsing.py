# Copyright 2023 Lei Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from hashlib import md5
from io import BytesIO
from typing import Any, List, Optional

import chardet
import docx2txt
import fitz
import pandas as pd
from langchain.docstore.document import Document

from langchain_lab import logger


class File(ABC):
    """Represents an uploaded file comprised of Documents"""

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
        return f"File(name={self.name}, id={self.id}, " " metadata={self.metadata}, docs={self.docs})"

    def __str__(self) -> str:
        return f"File(name={self.name}, id={self.id}, metadata={self.metadata})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name and self.id == other.id and self.metadata == other.metadata and self.docs == other.docs

    def __hash__(self):
        return hash((self.name, self.id, frozenset(self.metadata.items()), tuple(self.docs)))

    def copy(self) -> "File":
        """Create a deep copy of this File"""
        return self.__class__(
            name=self.name,
            id=self.id,
            metadata=deepcopy(self.metadata),
            docs=deepcopy(self.docs),
        )


def strip_consecutive_newlines(text: str) -> str:
    """Strips consecutive newlines from a string
    possibly with whitespace in between
    """
    return re.sub(r"\s*\n\s*", "\n", text)


def detect_big_bytesio_object(bytes_io_object: BytesIO):
    detector = chardet.UniversalDetector()
    for line in bytes_io_object:
        detector.feed(line)
        if detector.done:
            break
    detector.close()
    bytes_io_object.seek(0)
    encoding = detector.result["encoding"]
    logger.info(f"Detected file encoding: {encoding}")
    return encoding


class DocxFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "DocxFile":
        text = docx2txt.process(file)
        text = strip_consecutive_newlines(text)
        doc = Document(page_content=text.strip())
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])


class PdfFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "PdfFile":
        pdf = fitz.open(stream=file.read(), filetype="pdf")  # type: ignore
        docs = []
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            text = strip_consecutive_newlines(text)
            doc = Document(page_content=text.strip())
            doc.metadata["page"] = i + 1
            docs.append(doc)
        # file.read() mutates the file object, which can affect caching
        # so we need to reset the file pointer to the beginning
        file.seek(0)
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=docs)


class TxtFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "TxtFile":
        encoding = detect_big_bytesio_object(file)
        encodings = ["gb18030", "gb2312", "gbk"] if encoding.lower() == "gb2312" else [encoding]
        text = None
        for enc in encodings:
            try:
                logger.info(f"Trying to decode file with {enc}")
                text = file.read().decode(enc)
                break
            except Exception as e:
                logger.warn(f"decoding file with {enc} resulted in error: {e}")
        if text is not None:
            text = strip_consecutive_newlines(text)
            file.seek(0)
            doc = Document(page_content=text.strip())
            return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])
        else:
            raise Exception("Cannot read file")


class CsvFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "CsvFile":
        text = file.read().decode("utf-8")
        data = pd.read_csv(io.StringIO(text))
        docs = []
        for index, row in data.iterrows():
            line = {}
            for key in data.columns:
                line[key] = row[key].strip()
            doc = Document(page_content=json.dumps(line, ensure_ascii=False))
            docs.append(doc)
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=docs)


def read_file(file: BytesIO) -> File:
    """Reads an uploaded file and returns a File object"""
    if file.name.lower().endswith(".docx"):
        return DocxFile.from_bytes(file)
    elif file.name.lower().endswith(".pdf"):
        return PdfFile.from_bytes(file)
    elif file.name.lower().endswith(".txt"):
        return TxtFile.from_bytes(file)
    elif file.name.lower().endswith(".csv"):
        return CsvFile.from_bytes(file)
    else:
        raise NotImplementedError(f"File type {file.name.split('.')[-1]} not supported")
