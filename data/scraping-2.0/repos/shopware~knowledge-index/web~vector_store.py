import faiss
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from .cache import get_cache, set_cache
from .exception import EmptyEmbeddings

from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from .tracking import send_event
import asyncio

from .utils import safe_dir

# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/faiss.py
class FaissMap(VectorStore):
    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Dict[str, Document],
        index_to_docstore_id: Dict[int, str],
        stats: Any = None
    ):
        self.source_mapping = {}
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.stats = stats

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return []

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(embedding, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k, **kwargs)
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for j, i in enumerate(indices[0]):
            # This happens when not enough docs are returned.
            if i == -1:
                continue

            # edge case?
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.get(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            
            # filter out (hardcoded version filter)
            if (
                isinstance(kwargs, dict) and
                isinstance(kwargs.get('filter'), dict) and
                isinstance(kwargs['filter'].get('version'), str) and
                isinstance(doc.metadata.get('version'), str) and
                kwargs['filter'].get('version') != doc.metadata['version']
            ):
                continue
            
            docs.append((doc, scores[0][j]))

        # limit by context size
        finalDocs = []
        tokens = 0
        model = kwargs.get('model')
        if (model):
            print("Limiting to " + str(model.context))
            for doc in docs:
                docTokens = len(doc[0].page_content) / 4
                print("New doc has " + str(docTokens) + " tokens")

                if docTokens + tokens > model.context:
                    print("Break, limit exceeded")
                    break
                
                finalDocs.append(doc)
                tokens = tokens + docTokens
                print("New total tokens: " + str(tokens))

            return finalDocs

        return docs[:k]

    def similarity_search_by_id(self, id: str, k: int = 4) -> List[Document]:
        index_id = self.docstore[id].metadata["index_id"]
        embedding = self.index.reconstruct(index_id)
        return self.similarity_search_with_score_by_vector(embedding, k)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Any:
        #embeddings = embedding.embed_documents(texts)
        summary = get_cached_embeddings(embedding, texts, metadatas)
        reordered = summary["all"]
        texts = reordered["texts"]
        metadatas = reordered["metadatas"]
        embeddings = reordered["embeddings"]
            
        if len(embeddings) == 0:
            raise EmptyEmbeddings

        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = []
        assert len(texts) == len(metadatas)
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata["index_id"] = i
            assert "id" in metadata
            documents.append(Document(page_content=text, metadata=metadata))

        docstore = {doc.metadata["id"]: doc for doc in documents}

        index_to_id = {i: doc.metadata["id"] for i, doc in enumerate(documents)}

        cb = None
        if (isinstance(embedding, OpenAIEmbeddings)):
            cb = get_openai_callback()

        response = cls(embedding.embed_query, index, docstore, index_to_id, summary["new"])

        if (cb is not None):
            asyncio.run(send_event('all', 'from_texts', {**cb.__dict__}))
            print("Tracked")

        return response

    def save_local(self, folder_path: str) -> None:
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        index_faiss = safe_dir(folder_path, "index.faiss")
        index_pkl = safe_dir(folder_path, "index.pkl")

        # save index separately since it is not picklable
        faiss.write_index(self.index, index_faiss)

        # save docstore and index_to_docstore_id
        with open(index_pkl, "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(cls, folder_path: str, embeddings: Embeddings) -> Any:
        # load index separately since it is not picklable
        index = faiss.read_index(safe_dir(folder_path, "index.faiss"))

        # load docstore and index_to_docstore_id
        with open(safe_dir(folder_path, "index.pkl"), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings.embed_query, index, docstore, index_to_docstore_id)


def get_cached_embeddings(
        embedding: Embeddings,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
):
    cached = {
        "texts": [],
        "metadatas": [],
        "embeddings": []
    }
    non_cached = {
        "texts": [],
        "metadatas": [],
        "embeddings": []
    }

    # split by cached and non-cached
    for i, text in enumerate(texts):
        cached_embedding = get_cache(text)

        if cached_embedding:
            # mid-store texts, metadatas and embeddings
            cached["texts"].append(text)
            cached["metadatas"].append(metadatas[i])
            cached["embeddings"].append(cached_embedding)
        else:
            non_cached["texts"].append(text)
            non_cached["metadatas"].append(metadatas[i])
            # embeddings are calculated below

    # get non-existent embeddings
    non_cached["embeddings"] = embedding.embed_documents(non_cached["texts"])

    # cache and merge cached and new embeddings
    for i in range(len(non_cached["texts"])):
        text = non_cached["texts"][i]
        embedding = non_cached["embeddings"][i]
        set_cache(text, embedding)

        # merge
        cached["texts"].append(text)
        cached["metadatas"].append(non_cached["metadatas"][i])
        cached["embeddings"].append(embedding)

    return {
        "all": cached,
        "new": non_cached,
    }
