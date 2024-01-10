"""Functions to create the Vector database."""
import os
import pathlib
from dataclasses import dataclass

from typing import List, Union, Dict, Callable

from langchain.embeddings import base
from langchain import embeddings
from langchain.schema import Document
from langchain import vectorstores

import numpy as np
import yaml
import json
import streamlit as st

from .utils import get_device
from .download_transcripts import create_chunked_data
from .create_documents import extract_documents_from_list_of_dicts

DocumentDict = Dict[str, Union[str, float]]
PathLike = Union[str, os.PathLike]

MODEL_TYPES = {
    "sentence-transformers": embeddings.SentenceTransformerEmbeddings,
    "openai": embeddings.OpenAIEmbeddings,
}


@dataclass
class EmbeddingModelSpec:
    """Class to store the specification of an embedding model.

    Attributes:
        model_name: The name of the embedding model.
        model_type: The type of the embedding model. Can be
            `sentence-transformers` or `openai`.
        max_seq_length: The maximum number of tokens the model can handle.
    """
    model_name: str
    model_type: str
    max_seq_length: int

    def __post_init__(self):
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Model type {self.model_type} is not supported."
                             f" The supported model types are "
                             f"{list(MODEL_TYPES.keys())}.")


EMBEDDING_MODELS = [
    EmbeddingModelSpec(model_name="msmarco-MiniLM-L-6-v3",
                       model_type="sentence-transformers",
                       max_seq_length=512),
    EmbeddingModelSpec(model_name="msmarco-distilbert-base-v4",
                       model_type="sentence-transformers",
                       max_seq_length=512),
    EmbeddingModelSpec(model_name="msmarco-distilbert-base-tas-b",
                       model_type="sentence-transformers",
                       max_seq_length=512),
    EmbeddingModelSpec(model_name="text-embedding-ada-002",
                       model_type="openai",
                       max_seq_length=8191),
]

EMBEDDING_MODELS_NAMES = [embedding_model.model_name
                          for embedding_model in EMBEDDING_MODELS]


def get_embedding_model(embedding_model_name: str,
                        ) -> base.Embeddings:
    """Returns the embedding model.

    Args:
        embedding_model_name (str): The name of the embedding model.

    Raises:
        ValueError: If the model type is not supported.
    """
    embedding_model_spec = get_embedding_spec(embedding_model_name)
    if embedding_model_spec.model_type == "sentence-transformers":
        model_name = f"sentence-transformers/{embedding_model_spec.model_name}"
        device = get_device()
        model = embeddings.SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    elif embedding_model_spec.model_type == "openai":
        model = embeddings.OpenAIEmbeddings(  # type: ignore
            model=embedding_model_spec.model_name,
        )
    else:
        raise ValueError(f"Model type {embedding_model_spec.model_type} is not"
                         f" supported. The supported model types are "
                         f"{list(MODEL_TYPES.keys())}.")
    return model


def get_embedding_spec(model_name: str) -> EmbeddingModelSpec:
    """Returns the embedding model specification.

    Args:
        model_name (str): The name of the embedding model.

    Raises:
        ValueError: If the model name is not supported.
    """
    for embedding_model_spec in EMBEDDING_MODELS:
        if embedding_model_spec.model_name == model_name:
            return embedding_model_spec

    supported_model_names = [embedding_model_spec.model_name
                             for embedding_model_spec in EMBEDDING_MODELS]
    raise ValueError(f"Model name {model_name} is not supported. The "
                     f"supported model names are {supported_model_names}.")


def create_vectorstore(embedding_model_name: str,
                       documents: List[Document],
                       vector_store_type: str = "in-memory",
                       **kwargs) -> vectorstores.VectorStore:
    """Returns a vector store that contains the vectors of the documents.

    Currently, it only supports "in-memory" mode. In the future, it may
    support "chroma-db" mode as well.

    Note:
        In order to be able to make the vector store persistent, the
        `vector_store_type` should be `chroma-db` and the `kwargs` should
        contain the `persist_directory` argument with the path to the directory
        where the vector store will be saved or loaded from. The
        `persist_directory` is where Chroma will store its database files on
        disk, and load them on start.

    Args:
        embedding_model_name (str): The name of the embedding model.
        documents (List[Document]): List of documents.
        vector_store_type (str): The vector store type. Can be `chroma-db` or
            `in-memory`.
        **kwargs: Additional arguments passed to the `from_documents` method.

    Raises:
        ValueError: If the `persist_directory` argument is not provided when
            the vector store type is `chroma-db`.
    """

    if vector_store_type == "chroma-db" and "persist_directory" not in kwargs:
        raise ValueError(
            "The `persist_directory` argument should be provided when the "
            "vector store type is `chroma-db`. If you want to use an in-memory"
            " vector store, set the `vector_store_type` argument to "
            "`in-memory`.")

    object_mapper: Dict[str, Callable] = {
        # "chroma-db": vectorstores.Chroma.from_documents,
        "in-memory": vectorstores.DocArrayInMemorySearch.from_documents,
    }
    embedding_model = get_embedding_model(embedding_model_name)

    vectorstore = object_mapper[vector_store_type](
        documents, embedding_model, **kwargs
    )
    return vectorstore


def save_vectorstore(chroma_vectorstore: vectorstores.Chroma) -> None:
    """Makes the vectorstore persistent in the local disk.

    The vectorstore is saved in the persist directory indicated when the
    vectorstore was created.

    Args:
        chroma_vectorstore (VectorStore): The vectorstore.
    """
    chroma_vectorstore.persist()


def load_vectorstore(persist_directory: PathLike) -> vectorstores.Chroma:
    """Loads a vectorstore from the local disk.

    Args:
        persist_directory (Union[str, os.PathLike]): The directory where the
            vectorstore is saved.

    Returns:
        VectorStore: The Chroma vectorstore.
    """
    chroma_vectorstore = vectorstores.Chroma(
        persist_directory=str(persist_directory)
    )
    return chroma_vectorstore


def _create_hyperparams_yaml(directory: PathLike,
                             model_name: str,
                             max_chunk_size: int,
                             min_overlap_size: int):
    """Creates the hyperparams.yaml file in the directory."""
    hyperparams = {
        "model_name": model_name,
        "max_chunk_size": max_chunk_size,
        "min_overlap_size": min_overlap_size,
    }
    # Create the directory if it does not exist.
    pathlib.Path(directory).mkdir(parents=False, exist_ok=True)
    hyperparams_path = pathlib.Path(directory) / "hyperparams.yaml"
    with open(hyperparams_path, "w") as file:
        yaml.dump(hyperparams, file)


def load_hyperparams(directory: PathLike) -> Dict[str, Union[str, int]]:
    """Loads the hyperparams.yaml file in the directory."""
    hyperparams_path = pathlib.Path(directory) / "hyperparams.yaml"
    with open(hyperparams_path, "r") as file:
        hyperparams = yaml.load(file, Loader=yaml.FullLoader)
    return hyperparams


def save_json(chunked_data: List[dict],
              path: pathlib.Path,
              file_name: str) -> None:
    """Saves the data in a json file.

    Args:
        chunked_data (List[dict]): The data to be saved.
        path (PathLike): The path to the json file.
        file_name (str): The name of the json file.
    """
    # Create the directory if it does not exist.
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / file_name
    with open(file_path, "w") as file:
        json.dump(chunked_data, file)


def create_embeddings_pipeline(retriever_directory: PathLike,
                               embedding_model_name: str,
                               max_chunk_size: int,
                               min_overlap_size: int,
                               use_st_progress_bar: bool = True) -> None:
    """Sets up the embeddings for the given embedding model in the directory.

    Steps:
        1. Creates the retriever_directory if it does not exist.

        2. Creates the hyperparams.yaml file.

        3. Chunks the data.

        4. Creates the embeddings and saves them in the retriever_directory.

    Args:
        retriever_directory (PathLike): The directory where the embeddings will
            be saved. It should be inside a `data/playlist_name` directory.
            This function assumes that the playlist directory contains a
            `raw` directory with the json files of each video.
        embedding_model_name (str): The name of the embedding model.
        max_chunk_size (int): The maximum number of characters in a chunk.
        min_overlap_size (int): The minimum number of characters in the overlap
            between two consecutive chunks.
        use_st_progress_bar (bool): Whether to use the Streamlit progress bar
            or not.
    """
    retriever_directory = pathlib.Path(retriever_directory)
    embedding_model = get_embedding_model(embedding_model_name)

    # Create the hyperparams.yaml file.
    _create_hyperparams_yaml(
        retriever_directory,
        embedding_model_name,
        max_chunk_size,
        min_overlap_size
    )

    playlist_directory = pathlib.Path(retriever_directory).parent
    json_files_directory = playlist_directory / "raw"
    chunked_data_directory = retriever_directory / "chunked_data"
    json_files = list(json_files_directory.glob("*.json"))

    st_progress_bar = st.progress(0) if use_st_progress_bar else None
    total = len(json_files)

    # Create the `processed` directory if it does not exist.
    pathlib.Path(retriever_directory).mkdir(parents=True, exist_ok=True)

    for i, json_file_path in enumerate(json_files, start=1):
        if st_progress_bar is not None:
            st_progress_bar.progress(i / total, f"{i}/{total}")

        chunked_data = create_chunked_data(
            json_file_path,
            max_chunk_size,
            min_overlap_size
        )

        file_name = json_file_path.stem

        # Save the chunked data in the `processed` directory.
        save_json(chunked_data, chunked_data_directory, f"{file_name}.json")

        new_documents = extract_documents_from_list_of_dicts(chunked_data)
        documents_text = [document.page_content for document in new_documents]

        new_video_embeddings = embedding_model.embed_documents(documents_text)
        new_video_embeddings = np.array(new_video_embeddings)  # type: ignore

        # Save the embeddings in the `embeddings` directory.
        embeddings_directory = retriever_directory / "embeddings"
        # Create the directory if it does not exist.
        pathlib.Path(embeddings_directory).mkdir(exist_ok=True)
        embeddings_path = embeddings_directory / f"{file_name}.npy"
        np.save(str(embeddings_path), new_video_embeddings)


def load_embeddings(embedding_directory: PathLike) -> List[np.ndarray]:
    """Loads the embeddings from the retriever_directory.

    Args:
        embedding_directory (PathLike): The directory where the embeddings are
            saved.

    Returns:
        List[np.ndarray]: The embeddings. The order of the embeddings in
            the list is the same as the order of the json files in the
            `processed` directory.
    """

    numpy_files = list(pathlib.Path(embedding_directory).glob("*.npy"))

    video_embeddings = []
    for numpy_file in numpy_files:
        embedding = np.load(str(numpy_file))
        video_embeddings.append(embedding)

    return video_embeddings
