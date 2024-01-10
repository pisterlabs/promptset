import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import webvtt
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

VECTOR_STORE_PATH = Path("vectorstores")


class TextContent(ABC):
    def __init__(self, texts: List[str], metadata: List[dict], name: str) -> None:
        self.texts = texts
        self.metadata = metadata
        self.name = name
        self.vs = None

    @abstractmethod
    def _chunk(self):
        pass

    @abstractmethod
    def combine_docs(self, docs, document_separator="\n\n"):
        pass

    def vectorstore(self):
        chunked_texts, chunked_metadata = self._chunk()
        vs = create_vectorstore(self.name, chunked_texts, chunked_metadata)
        self.vs = vs
        return vs

    def retriever(self):
        if self.vs is None:
            self.vectorstore()
        return self.vs.as_retriever(search_kwargs={"k": 5})


class VTTContent(TextContent):
    def __init__(self, texts: List[str], metadata: List[dict], name: str) -> None:
        super().__init__(texts, metadata, name)

    def _chunk(self):
        chunk_size = 10
        n = len(self.texts)
        chunks = []
        chunked_metadata = []
        for i in range(0, n - chunk_size, chunk_size):
            chunk_captions = self.texts[i : i + chunk_size]
            chunks.append(" ".join(chunk_captions))

            chunked_metadata.append(
                {
                    "start_time": self.metadata[i]["start_time"],
                    "end_time": self.metadata[i + chunk_size - 1]["end_time"],
                    "doc_name": self.metadata[i]["doc_name"],
                }
            )
        return chunks, chunked_metadata

    def combine_docs(self, docs, document_separator="\n\n"):
        doc_strings = [
            f"Document Name: {doc.metadata['doc_name']}\nTimestamp: {doc.metadata['start_time']}\nContent: {doc.page_content}"
            for doc in docs
        ]
        return document_separator.join(doc_strings)

    @classmethod
    def from_path(cls, transcript_path: Path, name: str | None = None):
        if not transcript_path.exists():
            raise ValueError(f"Path {transcript_path} does not exist.")

        transcripts = transcript_path.glob("*.txt")

        texts = []
        metadata = []

        for transcript in transcripts:
            captions = webvtt.read(transcript)
            doc_name = transcript.stem

            caption_texts = []
            caption_metadata = []

            for caption in captions:
                meta = {
                    "start_time": caption.start,
                    "end_time": caption.end,
                    "doc_name": doc_name,
                }
                caption_metadata.append(meta)

                caption_texts.append(caption.text)

            texts.extend(caption_texts)
            metadata.extend(caption_metadata)

        name = name or transcript_path.stem
        return cls(texts, metadata, name)


class MarkdownContent(TextContent):
    def __init__(self, texts: List[str], metadata: List[dict], name: str) -> None:
        super().__init__(texts, metadata, name)

    def _chunk(self):
        chunk_size = 1000
        overlap = 0.5

        overlap = int(chunk_size * overlap)
        chunks = []
        start = 0
        end = chunk_size
        while start < len(self.texts[0]):
            chunks.append(self.texts[0][start:end])
            start += chunk_size - overlap
            end = start + chunk_size

        return chunks, [self.metadata[0] for _ in range(len(chunks))]

    def combine_docs(self, docs, document_separator="\n\n"):
        doc_strings = [f"Content: {doc.page_content}" for doc in docs]
        return document_separator.join(doc_strings)

    @classmethod
    def from_path(cls, markdown_path: Path):
        markdowns = markdown_path.glob("*.md")
        texts = []
        metadata = []
        for markdown in markdowns:
            doc_name = markdown.stem
            texts.append(markdown.read_text())
            metadata.append(
                {
                    "doc_name": doc_name,
                }
            )
        return cls(texts, metadata, markdown_path.stem)


def _exists_vectorstore(name: str):
    return os.path.exists(VECTOR_STORE_PATH / name)


def create_vectorstore(
    name: str,
    chunks: List[str],
    metadata: List[dict] = None,
    embedding=OpenAIEmbeddings(),
    vectorstore=FAISS,
):
    if _exists_vectorstore(name):
        print(
            f"The vectorstore {name} already exists. Change the name or delete it to generate a new one."
        )
        vectorstore = vectorstore.load_local(
            VECTOR_STORE_PATH / name, embeddings=embedding
        )
        return vectorstore
    elif chunks and metadata:
        vectorstore = vectorstore.from_texts(
            chunks, metadatas=metadata, embedding=embedding
        )
        vectorstore.save_local(VECTOR_STORE_PATH / name)
        return vectorstore
    elif chunks:
        vectorstore = vectorstore.from_texts(chunks, embedding=embedding)
        vectorstore.save_local(VECTOR_STORE_PATH / name)
        return vectorstore
    else:
        raise ValueError(
            "Either a vectorstore with the given name must exist or chunks and metadata must be provided."
        )
