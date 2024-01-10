import deeplake
from langchain.vectorstores import DeepLake, VectorStore

from folderchat.constants import DATA_PATH
from folderchat.io import clean_string_for_storing
from folderchat.loader import load_data_source, split_docs
from folderchat.logging import logger
from folderchat.models import MODES, get_embeddings


def get_dataset_path(data_source: str, options: dict, credentials: dict) -> str:
    dataset_name = clean_string_for_storing(data_source)
    # we need to differntiate between differently chunked datasets
    dataset_name += f"-{options['chunk_size']}-{options['chunk_overlap']}-{options['model'].embedding}"
    if options["mode"] == MODES.LOCAL:
        dataset_path = str(DATA_PATH / dataset_name)
    else:
        dataset_path = f"hub://{credentials['activeloop_org_name']}/{dataset_name}"
    return dataset_path


def get_vector_store(data_source: str, options: dict, credentials: dict) -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = get_embeddings(options, credentials)
    dataset_path = get_dataset_path(data_source, options, credentials)
    if deeplake.exists(dataset_path, token=credentials["activeloop_token"]):
        logger.info(f"Dataset '{dataset_path}' exists -> loading")
        vector_store = DeepLake(
            dataset_path=dataset_path,
            read_only=True,
            embedding_function=embeddings,
            token=credentials["activeloop_token"],
        )
    else:
        logger.info(f"Dataset '{dataset_path}' does not exist -> uploading")
        docs = load_and_split_data_from_source(data_source, options)
        vector_store = DeepLake.from_documents(
            docs,
            embeddings,
            dataset_path=dataset_path,
            token=credentials["activeloop_token"],
        )
    logger.info(f"Vector Store {dataset_path} loaded!")
    return vector_store

def load_and_split_data_from_source(data_source: str, options: dict)-> list:
    return split_docs(load_data_source(data_source), options)
