from io import BytesIO
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import numpy as np
import traceback
from typing import Any, Dict, List

from s3_api import download_object
from bedrock_api import get_bedrock_client, get_embeddings_client
from vector_db_api import pinecone_index_exists, create_pinecone_index, populate_pinecone_index


default_index_name = os.environ["DEFAULT_PINECONE_INDEX_NAME"]
default_api_key = os.environ["DEFAULT_PINECONE_API_KEY"]
default_environment = os.environ["DEFAULT_PINECONE_ENVIRONMENT"]
default_dimension = int(os.environ["DEFAULT_EMBEDDING_DIMENSION"])
default_metric = os.environ["DEFAULT_PINECONE_METRIC"]

first_document_sets_embedding_dimension = os.environ["FIRST_DOCUMENT_SETS_EMBEDDING_DIMENSION"] == 'True' or os.environ["FIRST_DOCUMENT_SETS_EMBEDDING_DIMENSION"]

local_temp_storage_root = '/tmp'

bedrock_client = None
embeddings_client = None


document_loader_constructors = {
    "pdf": lambda file_path: PyPDFLoader(file_path)
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_clients():
    global bedrock_client
    global embeddings_client

    if not bedrock_client:
        bedrock_client = get_bedrock_client()
    if not embeddings_client:
        embeddings_client = get_embeddings_client(bedrock_client)
    return bedrock_client, embeddings_client

    
def create_doc_chunks(local_file_path: str, metadata: Dict[str, Any], chunk_size=1000, chunk_overlap=100) -> List[Any]:
    extension = local_file_path.split('.')[1]
    doc_loader = document_loader_constructors.get(extension)(local_file_path)
    if doc_loader:
        document = doc_loader.load()
        for document_fragment in document:
            document_fragment.metadata = metadata
        logger.info(f'doc length: {len(document)}\n')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        doc_chunks = text_splitter.split_documents(document)
        logger.info(f'After the split we have {len(doc_chunks)} documents.')
        return doc_chunks


def sanitize_key(key):
    "sanitize key name to try and avoid naming issues on the /tmp file system"
    k = key.replace(' ', '')
    k = k.replace('+', '')
    k = k.replace('-', '')
    return k


def get_vector_dimension_from_document(bucket, key) -> int:
    _, embeddings_client = get_clients()
    
    local_file_path = f"{local_temp_storage_root}/{key}"
    metadata = dict(source=key)
    
    try:
        download_object(bucket, key, local_file_path)
        doc_chunks = create_doc_chunks(local_file_path, metadata)
        logger.info(f"First doc chunk details: {doc_chunks[0]}")
        logger.info("Get embedding vector and its dimension")
        sample_embedding = np.array(embeddings_client.embed_query(doc_chunks[0].page_content))
        dimension = sample_embedding.shape[0]
        logger.info(f"Got dimension: {dimension}")
        return dimension
        
    except Exception as e:
        logger.error(f"Failed to process contents of '{bucket}' / '{key}' due to: '{e}'")
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        raise e
        
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

    
def handle_document(bucket, key):
    _, embeddings_client = get_clients()

    sanitized_key = sanitize_key(key)
    local_file_path = f"{local_temp_storage_root}/{sanitized_key}"
    metadata = dict(source=key)  # note: 'source' matches what is expected when running inference with source attribution
    
    if not pinecone_index_exists(default_index_name, default_api_key, default_environment):
        if first_document_sets_embedding_dimension:
                        
            dimension = get_vector_dimension_from_document(bucket, key)
            create_pinecone_index(default_index_name, default_api_key, default_environment, dimension, default_metric, verbose=True)
            logger.info(f"'JIT' created Pinecone index named {default_index_name} with api_key={default_api_key} / environment={default_environment} / dimension={default_dimension} / metric={default_metric}")
        else:
            logger.error(f"Pinecone index {default_index_name} does not exist - make sure to invoke 'bootstrap' to create it before uploading documents to the landing zone S3 bucket")
        
    try:
        download_object(bucket, key, local_file_path)
        doc_chunks = create_doc_chunks(local_file_path, metadata)
        logger.info(f"First doc chunk details: {doc_chunks[0]}")
        logger.info("Start inserting doc chunks into Pinecone")
        populate_pinecone_index(doc_chunks, embeddings_client, default_index_name, verbose=False)
        logger.info("Inserting doc chunks into Pinecone completed")
    except Exception as e:
        logger.error(f"Failed to process contents of '{bucket}' / '{key}' due to: '{e}' - ignoring")
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            
        