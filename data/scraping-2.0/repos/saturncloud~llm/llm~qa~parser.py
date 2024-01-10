from __future__ import annotations
from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from datasets import Dataset
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TextSplitter

from llm.utils.types import StrEnum

logger = logging.getLogger(__name__)


DEFAULT_INDEX_NAME = "faiss_embeddings"


class DataFields(StrEnum):
    """
    Fields used in dataset parsing and document indexing/search
    """

    ID = "id"
    TEXT = "text"
    EMBEDDING = "embedding"

    # Extra fields for split documents
    DOC_ID = "doc_id"
    DOC_OFFSET = "doc_offset"


class DatasetParser:
    """
    Parse, split, and embed datasets to get them ready for semantic search.

    Pass multiple embedding devices to support multiprocessed embeddings.
    Each should be running on a different device (e.g. GPU1, GPU2,...)
    """

    def __init__(self, *embedding_devices: Embeddings) -> None:
        self.embedding_devices = embedding_devices

    def embedding(self, index: Optional[int] = None) -> Embeddings:
        index = index or 0
        if len(self.embedding_devices) <= index:
            raise ValueError(f"No embedding device at index {index}")
        return self.embedding_devices[index]

    def format(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        source_text_field: Optional[str] = None,
        source_id_field: Optional[str] = None,
        include_meta: Union[bool, str, List[str]] = True,
    ) -> Dataset:
        """
        Parse fields in the given dataset to match the expected format.
        Add UUIDs to each row if not given

        params:
            source_text_field: Text field to exract from the source dataset
            source_id_field: Use custom ID column instead of generating UUIDs
            include_meta: Filter columns that will be kept along with text and ID
        """

        def _parse_batch(
            batch: Dict[str, List],
            source_text_field: str,
            source_id_field: Optional[str] = None,
            include_meta: Optional[List[str]] = None,
        ) -> Dict:
            texts = batch.pop(source_text_field)
            metadata = {}

            if source_id_field or DataFields.ID in batch:
                doc_ids = batch.pop(source_id_field or DataFields.ID)
                doc_ids = [str(uid) for uid in doc_ids]
            else:
                doc_ids = [str(uuid4()) for _ in range(len(texts))]

            if include_meta:
                for key in include_meta:
                    metadata[key] = batch[key]

            return {
                DataFields.ID: doc_ids,
                DataFields.TEXT: texts,
                **metadata,
            }

        if source_text_field is None:
            source_text_field = DataFields.TEXT
        if isinstance(include_meta, bool):
            include_meta = dataset.column_names if include_meta else []
            include_meta = [
                f for f in include_meta if f not in {source_text_field, source_id_field}
            ]
        elif isinstance(include_meta, str):
            include_meta = [include_meta]

        logger.info("Parsing dataset")
        return dataset.map(
            _parse_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "source_text_field": source_text_field,
                "source_id_field": source_id_field,
                "include_meta": include_meta,
            },
        )

    def split(
        self,
        dataset: Dataset,
        splitter: TextSplitter,
        batch_size: int = 100,
    ) -> Dataset:
        """
        Split the text in the given dataset into chunks, copying metadata for each split
        and tracking the offset from the parent document.

        Map ID -> DOC_ID, and assign a new ID to each split
        """

        def _split_batch(
            batch: Dict[str, List],
        ) -> Dict:
            texts = batch.pop(DataFields.TEXT)
            split_texts = []
            keys = [DataFields.ID, DataFields.DOC_ID, DataFields.DOC_OFFSET, *batch.keys()]
            split_data: Dict[str, List[Any]] = {key: [] for key in keys}

            for i, text in enumerate(texts):
                splits = splitter.split_text(text)
                split_texts.extend(splits)

                # Duplicate other data for each split, generate IDs, and track offset
                for j in range(len(splits)):
                    split_data[DataFields.ID].append(str(uuid4()))
                    split_data[DataFields.DOC_OFFSET].append(j)
                    for key, vals in batch.items():
                        if key == DataFields.ID:
                            key = DataFields.DOC_ID
                        split_data[key].append(deepcopy(vals[i]))

            return {
                DataFields.TEXT: split_texts,
                **split_data,
            }

        self._validate(
            dataset, exclude=[DataFields.EMBEDDING, DataFields.DOC_OFFSET, DataFields.DOC_ID]
        )
        logger.info("Splitting dataset")
        dataset = dataset.map(
            _split_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
        )
        return dataset

    def embed(
        self,
        dataset: Dataset,
        batch_size: int = 100,
    ) -> Dataset:
        """
        Compute semantic embeddings for the text field and add to the dataset
        """

        def _embed_batch(batch: Dict[str, List], rank: Optional[int]) -> Dict:
            texts = batch[DataFields.TEXT]
            embeddings = self.embedding(rank).embed_documents(texts)
            return {
                **batch,
                DataFields.EMBEDDING: embeddings,
            }

        self._validate(dataset, exclude=[DataFields.EMBEDDING])
        logger.info("Embedding dataset")
        return dataset.map(
            _embed_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=len(self.embedding_devices),
            with_rank=True,
        )

    def index(
        self,
        dataset: Dataset,
        index_name: str = DEFAULT_INDEX_NAME,
        index_path: Optional[str] = None,
        **kwargs,
    ) -> Dataset:
        """
        Add FAISS index on embeddings column for basic semantic searchs
        """
        dataset = dataset.add_faiss_index(str(DataFields.EMBEDDING), index_name)
        if index_path:
            dataset.save_faiss_index(index_name, index_path, **kwargs)
        return dataset

    def create_dataset(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        uuids: Optional[List[str]] = None,
        splitter: Optional[TextSplitter] = None,
        batch_size: int = 100,
        **kwargs,
    ) -> Dataset:
        """
        Combine data into a document dataset
        """
        rows: List[Dict[str, Any]] = [{}] * len(texts)
        for i, text in enumerate(texts):
            data = rows[i]
            data[DataFields.ID] = uuids[i] if uuids else str(uuid4())
            data[DataFields.TEXT] = text
            if metadatas:
                data.update(metadatas[i])
        dataset = Dataset.from_list(rows, **kwargs)

        dataset = self.format(
            dataset,
            batch_size=batch_size,
            source_text_field=str(DataFields.TEXT),
            source_id_field=str(DataFields.ID),
        )
        if splitter:
            dataset = self.split(dataset, splitter, batch_size=batch_size)
        return self.embed(dataset, batch_size=batch_size)

    def _validate(self, dataset: Dataset, exclude: Optional[List[str]] = None):
        given = set(dataset.column_names)
        expected = set(DataFields.values())
        if exclude:
            for field in exclude:
                expected.discard(field)

        for field in expected:
            if field not in given:
                raise ValueError(f"Invalid dataset. Expected fields: {expected}. Given: {given}")
