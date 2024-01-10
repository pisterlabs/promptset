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
from abc import ABC

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.langchain_lab.core.parsing import File


class Chunking(ABC):

    def __init__(self, chunk_size: int, chunk_overlap: int = 0, model_name="gpt-3.5-turbo"):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, file: File):
        chunked_docs = []
        for doc in file.docs:
            chunks = self.splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page": doc.metadata.get("page", 1),
                        "chunk": i + 1,
                        "source": f"{doc.metadata.get('page', 1)}-{i + 1}",
                    },
                )
                chunked_docs.append(doc)
        chunked_file = file.copy()
        chunked_file.docs = chunked_docs
        return chunked_file


def chunk_file(file: File, chunk_size: int, chunk_overlap: int = 0, model_name="gpt-3.5-turbo") -> File:
    """Chunks each document in a file into smaller documents
    according to the specified chunk size and overlap
    where the size is determined by the number of token for the specified model.
    """

    # split each document into chunks
    chunked_docs = []
    for doc in file.docs:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata.get("page", 1),
                    "chunk": i + 1,
                    "source": f"{doc.metadata.get('page', 1)}-{i + 1}",
                },
            )
            chunked_docs.append(doc)

    chunked_file = file.copy()
    chunked_file.docs = chunked_docs
    return chunked_file
