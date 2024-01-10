from __future__ import annotations

import re
import warnings
from typing import Optional

import pandas as pd
from context import download_embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.storage import BaseStore
from langchain.storage.in_memory import InMemoryStore
from langchain.vectorstores.faiss import FAISS


class MultiVectorFleetRetriever(MultiVectorRetriever):
    """A class to create retrievers from `fleet-context` embeddings."""

    @staticmethod
    def _prep_df(df: pd.DataFrame, library_name: str):
        def _join_metadata(_df: pd.DataFrame):
            return _df.join(
                _df["metadata"].apply(pd.Series),
                lsuffix="_orig",
                rsuffix="_md",
            )

        return df.assign(
            metadata=lambda _df: _df.metadata.apply(
                lambda md: {**md, "library_name": library_name},
            ),
        ).pipe(_join_metadata)

    @staticmethod
    def _get_vectorstore(joined_df: pd.DataFrame, **kwargs) -> FAISS:
        """Get FAISS vectorstore from joined df."""
        return FAISS.from_embeddings(
            joined_df[["text", "dense_embeddings"]].values,
            OpenAIEmbeddings(model="text-embedding-ada-002"),
            metadatas=joined_df["metadata"].tolist(),
            **kwargs,
        )

    @staticmethod
    def _df_to_parent_docs(joined_df: pd.DataFrame, sep: str = "\n") -> list[Document]:
        return (
            joined_df[["parent", "title", "text", "type", "url", "section_index"]]
            .rename(columns={"parent": "id"})
            .sort_values(["id", "section_index"])
            .groupby("id")
            .apply(
                lambda chunk: Document(
                    page_content=chunk.iloc[0]["title"]
                    + "\n"
                    + chunk["text"].str.cat(sep=sep),
                    metadata=chunk.iloc[0][["title", "type", "url", "id"]].to_dict(),
                ),
            )
            .tolist()
        )

    def __init__(
        self,
        df: pd.DataFrame,
        library_name: str,
        docstore: Optional[BaseStore] = None,
        parent_doc_sep: str = "\n",
        vectorstore_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        joined_df = self._prep_df(df, library_name)

        parent_docs = self._df_to_parent_docs(joined_df, sep=parent_doc_sep)

        vectorstore_kwargs = vectorstore_kwargs or {}
        vectorstore = self._get_vectorstore(joined_df, **vectorstore_kwargs)

        docstore = docstore or InMemoryStore()
        docstore.mset([(doc.metadata["id"], doc) for doc in parent_docs])

        super().__init__(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="parent",
            **kwargs,
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        library_name: str,
        **kwargs,
    ) -> MultiVectorFleetRetriever:
        """Create MultiVectorFleetRetriever from df."""
        return cls(df, library_name=library_name, **kwargs)

    @classmethod
    def from_library(
        cls,
        library_name: str,
        download_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> MultiVectorFleetRetriever:
        """Create MultiVectorFleetRetriever from library_name."""
        download_kwargs = download_kwargs or {}
        try:
            library_df = download_embeddings(library_name, **download_kwargs)
        except TypeError:
            if download_kwargs:
                warnings.warn(
                    "`download_kwargs` not yet implemented in `context`; ignoring.",
                )
            library_df = download_embeddings(library_name)
        return cls(library_df, library_name=library_name, **kwargs)

    @staticmethod
    def get_library_name_from_filename(filename: str) -> str:
        filename_pat = re.compile("libraries_(.*).parquet")

        search_result = filename_pat.search(filename)
        if search_result is None:
            raise ValueError(
                f"filename {filename} does not match pattern {filename_pat}",
            )
        return search_result.group(1)

    @classmethod
    def from_parquet(cls, filename: str, **kwargs) -> MultiVectorFleetRetriever:
        """Create MultiVectorFleetRetriever from parquet filename."""
        library_name = cls.get_library_name_from_filename(filename)
        return cls(pd.read_parquet(filename), library_name=library_name, **kwargs)
