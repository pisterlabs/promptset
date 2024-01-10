#!/usr/bin/env python3
import hashlib
import time
from typing import Any

import click
import langchain
from beartype import beartype
from click import Context
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma, VectorStore, Qdrant
from pycomfort.config import load_environment_keys
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.huggingface import *
from qdrant_client.http.models import PayloadSchemaType
from indexpaper.resolvers import *
from indexpaper.splitting import SourceTextSplitter, papers_to_documents, paginated_paper_to_documents
from indexpaper.utils import timing


def generate_id_from_data(data):
    """
    function to avoid duplicates
    :param data:
    :return:
    """
    if isinstance(data, str):  # check if data is a string
        data = data.encode('utf-8')  # encode the string into bytes
    return str(hex(int.from_bytes(hashlib.sha256(data).digest()[:32], 'little')))[-32:]

def db_with_texts(db: VectorStore, texts: list[str],
                  splitter: TextSplitter, id_field: Optional[str] = None, debug: bool = False):
    return db_with_documents(db, texts_to_documents(texts), splitter, id_field, debug)


def db_with_documents(db: VectorStore, documents: list[Document],
                      splitter: TextSplitter,
                      id_field: Optional[str] = None, debug: bool = False):
    """
    Function to add documents to a Chroma database.

    Args:
        db (Chroma): The database to add the documents to.
        documents (list[Document]): The list of documents to add.
        splitter (TextSplitter): The TextSplitter to use for splitting the documents.
        debug (bool): If True, print debug information. Default is False.
        id_field (Optional[str]): If provided, use this field from the document metadata as the ID. Default is None.

    Returns:
        Chroma: The updated database.
    """
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [doc.metadata[id_field] for doc in docs] if id_field is not None else None
    if debug:
        for doc in documents:
            logger.trace(f"ADD TEXT: {doc.page_content}")
            logger.trace(f"ADD METADATA {doc.metadata}")
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return db

@beartype
def init_qdrant(collection_name: str,
                path_or_url: Optional[str],
                embeddings: Optional[Embeddings],
                always_recreate: bool = True,
                api_key: Optional[str] = None,
                distance_func: str = "Cosine",
                prefer_grpc: bool = False,
                timeout: Optional[int] = 3600,
                indexes: Optional[dict[str, PayloadSchemaType]] = None
                ):
    """
    Function that initializes QDrant
    :param collection_name:
    :param path_or_url:
    :param embedding_function:
    :param api_key:
    :param distance_func:
    :param prefer_grpc:
    :return:
    """
    is_url = "ttp:" in path_or_url or "ttps:" in path_or_url
    path: Optional[str] = None if is_url else path_or_url
    url: Optional[str] = path_or_url if is_url else None
    logger.info(f"initializing quadrant database at {path_or_url}")
    client: QdrantClient = QdrantClient(
        url=url,
        port=6333,
        grpc_port=6334,
        prefer_grpc=is_url if prefer_grpc is None else prefer_grpc,
        api_key=api_key,
        path=path
    )
    from qdrant_client.http import models as rest
    #client.recreate_collection(collection_name)
    # Just do a single quick embedding to get vector size
    collections = client.get_collections()
    if always_recreate or not seq(collections.collections).exists(lambda c: c.name == collection_name):
        partial_embeddings = embeddings.embed_documents(["Hello world text!"])
        vector_size = len(partial_embeddings[0])
        logger.info(f"creating collection {collection_name},\n computed probe vector size for the model was {vector_size}")
        distance_func = distance_func.upper()
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance[distance_func]
            ), timeout=timeout
        )
        for k, v in indexes.items():
            client.create_payload_index(collection_name, k, v)
    return Qdrant(client, collection_name=collection_name, embeddings=embeddings)


def write_remote_db(url: str,
                    collection_name: str,
                    documents: list[Document],
                    splitter: TextSplitter,
                    id_field: Optional[str] = None,
                    embeddings: Optional[Embeddings] = None,
                    database: VectorDatabase = VectorDatabase.Qdrant,
                    key: Optional[str] = None,
                    prefer_grpc: Optional[bool] = False,
                    always_recreate: bool = False) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    if database == VectorDatabase.Qdrant:
        logger.info(f"writing a collection {collection_name} of {len(documents)} documents to quadrant db at {url}")
        start_time = time.perf_counter()
        api_key = os.getenv("QDRANT_KEY") if key == "QDRANT_KEY" or key == "key" else key
        db = init_qdrant(collection_name, path_or_url=url, embeddings=embeddings, api_key=api_key, prefer_grpc=prefer_grpc, always_recreate=always_recreate)
        db_updated = db_with_documents(db, documents, splitter,  id_field)
        end_time = time.perf_counter()
        execution_time: float = end_time - start_time
        return db_updated, url, execution_time
    else:
        raise Exception(f"Remote Chroma is not yet supported by this script!")
    pass


@beartype
def make_local_db(collection_name: str,
                  documents: list[Document],
                  splitter: TextSplitter,
                  embeddings: Optional[Embeddings] = None,
                  database: VectorDatabase = VectorDatabase.Chroma,
                  persist_directory: Optional[Path] = None,
                  id_field: Optional[str] = None,
                  prefer_grpc: Optional[bool] = False,
                  always_recreate: bool = False
                  ) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    """
    :param collection_name:
    :param documents:
    :param splitter:
    :param embeddings:
    :param database:
    :param persist_directory:
    :param id_field:
    :param prefer_grpc:
    :return:
    """
    start_time = time.perf_counter()

    # If no embeddings were provided, default to OpenAIEmbeddings
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    # Create the directory where the database will be saved, if it doesn't already exist
    if persist_directory is not None:
        where = persist_directory / collection_name
        where.mkdir(exist_ok=True, parents=True)
        where_str = str(where)
    else:
        where = None
        where_str = None

    # Create a Chroma database with the specified collection name and embeddings, and save it in the specified directory
    if database == VectorDatabase.Qdrant:
        db = init_qdrant(collection_name, where_str, embedding_function=embeddings,  prefer_grpc = prefer_grpc, always_recreate = always_recreate)
    else:
        db = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings)
    db_updated = db_with_documents(db, documents, splitter,  id_field)

    # Persist the changes to the database
    if persist_directory is not None:
        db_updated.persist()
    end_time = time.perf_counter()
    execution_time: float = end_time - start_time

    # Return the directory where the database was saved
    return db_updated, where, execution_time


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    # if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass


@beartype
def texts_to_documents(texts: list[str]) -> list[Document]:
    return [Document(
        page_content=text,
        metadata={"text": text}
    ) for text in texts]

@timing("indexing selected")
def index_selected_documents(documents: list[Document],
                          collection: str,
                          splitter: SourceTextSplitter,
                          embedding_type: EmbeddingType,
                          url: str,
                          key: str,
                          database: VectorDatabase = VectorDatabase.Chroma.value,
                          model: Optional[Union[Path, str]] = None,
                          prefer_grpc: Optional[bool] = None,
                          device: Device = Device.cpu
                          ) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    openai_key = load_environment_keys() #for openai key
    embeddings_function = resolve_embeddings(embedding_type, model, device)
    logger.info(f"embeddings are {embedding_type}")
    return write_remote_db(url, collection, documents, splitter, embeddings=embeddings_function, database=database, key=key, prefer_grpc=prefer_grpc)


@beartype
def index_selected_papers(folder_or_texts: Union[Path, list[str]],
                          collection: str,
                          splitter: SourceTextSplitter,
                          embedding_type: EmbeddingType,
                          include_meta: bool = True,
                          folder: Optional[Union[Path,str]] = None,
                          url: Optional[str] = None,
                          key: Optional[str] = None,
                          database: VectorDatabase = VectorDatabase.Chroma.value,
                          model: Optional[Union[Path, str]] = None,
                          prefer_grpc: Optional[bool] = None,
                          always_recreate: bool = False,
                          device: Device = Device.cpu
                          ) -> (Union[VectorStore, Any, langchain.vectorstores.Chroma], Optional[Union[Path, str]], float):
    openai_key = load_environment_keys() #for openai key
    embeddings_function = resolve_embeddings(embedding_type, model, device)
    logger.info(f"embeddings are {embedding_type}")
    documents = papers_to_documents(folder_or_texts, include_meta=include_meta) if isinstance(folder_or_texts, Path) else texts_to_documents(folder_or_texts)
    if url is not None or key is not None:
        return write_remote_db(url, collection, documents, splitter, embeddings=embeddings_function, database=database, key=key, prefer_grpc = prefer_grpc, always_recreate = always_recreate)
    else:
        if folder is None:
            logger.warning(f"neither url not folder are set, trying in memory")
            return make_local_db(collection, documents, splitter, embeddings_function, prefer_grpc = prefer_grpc, database=database, always_recreate = always_recreate)
        else:
            index = Path(folder) if isinstance(folder, str) else folder
            index.mkdir(exist_ok=True)
            where = index / f"{embedding_type.value}_{splitter.chunk_size}_chunk"
            where.mkdir(exist_ok=True, parents=True)
            logger.info(f"writing index of papers to {where}")
            return make_local_db(collection, documents, splitter, embeddings_function, persist_directory=where,  prefer_grpc = prefer_grpc, database=database, always_recreate = always_recreate)


def fast_index_papers(folder: Path, collection: str, url: Optional[str], key: Optional[str], model: str = EmbeddingModels.default.value, prefer_grpc: bool = True, parallel: Optional[int] = None, rewrite: bool = False, paginated: bool = True) -> Path:
    load_environment_keys(usecwd=True)
    assert not (url is None and key is None), "either database url or api_key should be provided!"
    chunk_size: int = 512
    #logger.info(f"computing embeddings into collection {collection} for {dataset} with model {model} using slices of {slice} starting from {start} with chunks of {chunk_size} tokens when splitting")
    splitter = HuggingFaceSplitter(model, tokens=chunk_size)
    api_key = os.getenv("QDRANT_KEY") if key == "QDRANT_KEY" or key == "key" else key
    is_url = "ttp:" in url or "ttps:" in url
    path: Optional[str] = None if is_url else url #actually the user can give either path or url
    url: Optional[str] = url if is_url else None
    logger.info(f"initializing quadrant database at {url}")
    client: QdrantClient = QdrantClient(
        url=url,
        port=6333,
        grpc_port=6334,
        prefer_grpc=is_url if prefer_grpc is None else prefer_grpc,
        api_key=api_key,
        path=path
    )
    client.set_model(model)
    collections = client.get_collections()
    if rewrite or not seq(collections.collections).exists(lambda c: c.name == collection):
        logger.info(f"creating collection {collection}")
        client.recreate_collection(collection_name=collection, vectors_config=client.get_fastembed_vector_params(on_disk=True))
        indexes: dict[str, PayloadSchemaType] = {
            "doi": PayloadSchemaType.TEXT,
            "source": PayloadSchemaType.TEXT,
            "document": PayloadSchemaType.TEXT
        }
        for k, v in indexes.items():
            client.create_payload_index(collection, k, v)
    docs = paginated_paper_to_documents(folder) if paginated else papers_to_documents(folder)
    texts = [d.page_content for d in docs]
    for d in docs:
        if "metadata" not in d.metadata: #ugly fix for metadata issue
            d.metadata["metadata"] = d.metadata.copy()
    metadatas = [d.metadata for d in docs]
    ids = [generate_id_from_data(d.page_content) for d in docs]
    client.add(
        collection_name=collection,
        documents=texts,
        metadata=metadatas,
        ids=ids,
        parallel=parallel
    )
    logger.info(f"added {len(texts)} documents")
    return client
    #paper_set.fast_index_by_slice(n = slice, client=client, collection_name=collection, batch_size=batch_size, parallel=parallel)