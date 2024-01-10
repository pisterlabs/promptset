import abc
import dataclasses
import os
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from rich import print

import peachdb.embedder.models.base
from peachdb.embedder.models.multimodal_imagebind import ImageBindModel
from peachdb.embedder.models.sentence_transformer import SentenceTransformerModel
from peachdb.embedder.openai_ada import OpenAIAdaEmbedder
from peachdb.embedder.utils import Modality, S3File, S3Folder, is_s3_uri


@dataclasses.dataclass
class BackendConfig:
    embedding_generator: str
    distance_metric: str
    embeddings_dir: str
    metadata_path: str
    id_column_name: str
    modality: Modality


class BackendBase(abc.ABC):
    def __init__(
        self,
        backend_config: BackendConfig,
    ):
        # TODO: refactor below to clean up
        embeddings_dir = backend_config.embeddings_dir
        metadata_path = backend_config.metadata_path
        embedding_generator = backend_config.embedding_generator
        distance_metric = backend_config.distance_metric
        id_column_name = backend_config.id_column_name
        modality = backend_config.modality
        self._distance_metric = distance_metric
        self._id_column_name = id_column_name
        self._metadata_filepath = self._get_metadata_filepath(metadata_path)
        self._modality = modality
        self._embedding_generator = embedding_generator

        self._embeddings, self._ids = self._get_embeddings(embeddings_dir)
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("Duplicate ids found in the embeddings file.")

        if self._embedding_generator == "sentence_transformer_L12":
            self._encoder: peachdb.embedder.models.base.BaseModel = SentenceTransformerModel()
        elif self._embedding_generator == "imagebind":
            self._encoder = ImageBindModel()
        elif self._embedding_generator == "openai_ada":
            self._openai_encoder = OpenAIAdaEmbedder()
        else:
            raise ValueError(f"Unknown embedding generator: {embedding_generator}")

    @abc.abstractmethod
    def _process_query(self, query_embedding, top_k: int = 5) -> tuple:
        pass

    def process_query(self, query: str, modality: Modality, top_k: int = 5) -> tuple:
        print("Embedding query...")
        if self._embedding_generator == "openai_ada":
            assert modality == Modality.TEXT
            query_embedding = np.asarray(self._openai_encoder.calculate_embeddings([query])[0:1])
            return self._process_query(query_embedding, top_k)
        else:
            if modality == Modality.TEXT:
                query_embedding = self._encoder.encode_texts(texts=[query], batch_size=1, show_progress_bar=True)
            elif modality == Modality.AUDIO:
                query_embedding = self._encoder.encode_audio(local_paths=[query], batch_size=1, show_progress_bar=True)
            elif modality == Modality.IMAGE:
                query_embedding = self._encoder.encode_image(local_paths=[query], batch_size=1, show_progress_bar=True)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            return self._process_query(query_embedding, top_k)

    def fetch_metadata(self, ids, namespace: Optional[str]) -> pd.DataFrame:
        print("Fetching metadata...")

        # NOTE: this is a hack, as we keep updating the metadata.
        data = duckdb.read_csv(self._metadata_filepath, header=True)
        id_str = " OR ".join([f"{self._id_column_name} = '{id}'" for id in ids])
        if namespace is None:
            metadata = duckdb.sql(f"SELECT * FROM data WHERE {id_str}").df()
        else:
            metadata = duckdb.sql(f"SELECT * FROM data WHERE ({id_str}) AND (namespace = '{namespace}')").df()

        return metadata

    def _get_embeddings(self, embeddings_dir: str):
        if not is_s3_uri(embeddings_dir):
            return self._load_embeddings(embeddings_dir)

        print("[bold]Downloading calculated embeddings...[/bold]")
        with S3Folder(embeddings_dir) as tmp_local_embeddings_dir:
            return self._load_embeddings(tmp_local_embeddings_dir)

    def _load_embeddings(self, embeddings_dir: str) -> tuple:
        """Loads and preprocesses the embeddings from a parquet file."""
        assert os.path.exists(embeddings_dir)

        print("[bold]Loading embeddings from parquet file...[/bold]")
        df = pd.read_parquet(embeddings_dir, "pyarrow")

        print("[bold]Converting embeddings to numpy array...[/bold]")
        if self._modality == Modality.TEXT:
            # TODO: these keys name are used in embedder.containers.base, so we should refactor
            embeddings = np.array(df["text_embeddings"].values.tolist()).astype("float32")
        elif self._modality == Modality.AUDIO:
            embeddings = np.array(df["audio_embeddings"].values.tolist()).astype("float32")
        elif self._modality == Modality.IMAGE:
            embeddings = np.array(df["image_embeddings"].values.tolist()).astype("float32")
        else:
            raise ValueError(f"Unknown modality: {self._modality}")
        ids = np.asarray(df["ids"].apply(str).values.tolist())
        return embeddings, ids

    def _get_metadata_filepath(self, metadata_path: str) -> str:
        if not is_s3_uri(metadata_path):
            return metadata_path

        print("[bold]Downloading metadata file...[/bold]")
        self._metadata_fileref = S3File(metadata_path)
        return self._metadata_fileref.download()

    def cleanup(self):
        if is_s3_uri(self._metadata_path):
            self._metadata_fileref.cleanup()

        if is_s3_uri(self._embeddings_dir):
            self._embeddings_dir.cleanup()
