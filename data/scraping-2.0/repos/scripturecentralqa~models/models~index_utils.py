"""Utilities for indexing documents in Pinecone."""
import hashlib
import time
from typing import Any
from typing import List
from typing import cast

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from tqdm.auto import tqdm

from models.split_utils import get_voyageai_embedder
from models.split_utils import remove_markdown


def get_doc_id(doc: Document) -> str:
    """Get a unique ID for a document."""
    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()
    # hash the url (if it exists) + text
    text = doc.metadata.get("url", "") + "~" + doc.page_content
    # Update the hash object with the text's bytes
    sha256.update(text.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    return sha256.hexdigest()


class VoyageAIEmbedder(Embeddings):
    """Voyage AI Embedder."""

    def __init__(self, model: str = "voyage-01", input_type: str = "document"):
        """Initialize."""
        self.embedder = get_voyageai_embedder(model, input_type)

    def embed_documents(self, paragraphs: list[str]) -> list[list[float]]:
        """Get Voyage AI embeddings for paragraphs."""
        return cast(list[list[float]], [emb.tolist() for emb in self.embedder(paragraphs)])

    def embed_query(self, text: str) -> List[float]:
        """Get Voyage AI embeddings for a query."""
        return cast(list[float], self.embedder([text])[0].tolist())

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get asynchronous Voyage AI embeddings for paragraphs, not implemented."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement aembed_documents()")

    async def aembed_query(self, text: str) -> List[float]:
        """Get asynchronous Voyage AI embeddings for a query, not implemented."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement aembed_query()")


def embed_documents(embedder: Any, docs: list[Document], batch_size: int = 100, delay: float = 1.0) -> list[Any]:
    """Return document embeddings using an embedder."""
    embeddings = []
    for i in tqdm(range(0, len(docs), batch_size)):
        # wait between batches
        time.sleep(delay)
        # set end position of batch
        i_end = min(i + batch_size, len(docs))
        text_batch = [
            remove_markdown(f"{doc.metadata['title']}\n\n{doc.page_content}").strip() for doc in docs[i:i_end]
        ]
        embed_batch = embedder.embed_documents(text_batch)
        embeddings.extend(embed_batch)
    return embeddings


def index_documents(index: Any, embeddings: list[Any], docs: list[Document], batch_size: int = 100) -> None:
    """Index documents in Pinecone."""
    for i in tqdm(range(0, len(docs), batch_size)):
        # set end position of batch
        i_end = min(i + batch_size, len(docs))
        # get batch of lines and IDs
        metadata_batch = [
            dict(
                docs[n].metadata,
                text=docs[n].page_content,
                source=get_doc_id(docs[n]),
            )
            for n in range(i, i_end)
        ]
        ids_batch = [get_doc_id(doc) for doc in docs[i:i_end]]
        # get embeddings
        embed_batch = embeddings[i:i_end]
        # create upsert batch
        to_upsert = zip(ids_batch, embed_batch, metadata_batch, strict=True)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))
