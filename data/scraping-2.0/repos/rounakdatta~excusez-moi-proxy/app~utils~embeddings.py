import openai
import hashlib
import numpy
from db.db import Database
from utils.common import base64_encode_string, count_embedding_tokens
from config.openai import openai_config
import numpy
from typing import List


async def generate_id_for_embedding(url: str, payload: str):
    """
    Logic to compute embedding_id given a particular URL and its content
    """
    # generate SHA256 digest for the url
    urlSha = hashlib.sha256(url.encode('utf-8'))
    urlShaDigest = urlSha.hexdigest()[:5]

    # generate SHA256 digest for the payload
    payloadSha = hashlib.sha256(payload.encode('utf-8'))
    payloadShaDigest = payloadSha.hexdigest()[:10]

    # embedding id the combination of the two, separated by colon
    return ":".join([urlShaDigest, payloadShaDigest])


async def check_if_embeddings_already_generated(db: Database, embedding_id: str):
    """
    Checks in database if embeddings have been already been generated for a particular id
    """
    return await db.fetch_val("SELECT EXISTS(SELECT 1 FROM embeddings WHERE embedding_id = $1)", embedding_id)


async def get_if_embeddings_already_generated(db: Database, embedding_id: str):
    """
    Returns the already generated embeddings for a particular id
    """
    return await db.fetch_all("SELECT * FROM embeddings WHERE embedding_id = $1", embedding_id)


async def break_down_document(payload: str, sentence_splitter):
    """
    Utility to break down the complete document into smaller chunks to be batch-sent for embeddings generation.
    Smaller chunks also help us reduce surface area while querying
    """
    total_payload_token_count = await count_embedding_tokens(payload)
    n_of_splits = total_payload_token_count // int(
        openai_config.embedding_model_optimal_input_tokens) + 1

    # we break it heuristically by text, so token count for each piece wouldn't be equal
    return sentence_splitter.split_text(payload)


async def generate_embeddings_external(payload):
    """
    Use external API (OpenAI) to generate the embeddings for a given payload.
    Here payload could be an array or a single element.
    The embeddings are then returned as a numpy array
    """
    external_response = openai.Embedding.create(
        model=openai_config.embedding_model_name, input=payload)
    return [numpy.array(el["embedding"]) for el in external_response["data"]]


async def persist_embeddings_to_storage(db: Database, embeddings: List[numpy.ndarray], raw_collection: List[str], url: str, embedding_id: str, content_type: str):
    """
    Persist computed embeddings to database, so that they can be later retrieved
    """
    embedding_data_rows = []

    for i in range(len(raw_collection)):
        embedding_data = {
            "embedding": embeddings[i],
            "raw_payload": await base64_encode_string(raw_collection[i]),
            "anchor_url": await base64_encode_string(url),
            "embedding_id": embedding_id,
            "content_type": content_type
        }
        embedding_data_row = tuple(embedding_data.values())
        embedding_data_rows.append(embedding_data_row)

    await db.execute_with_vector_registered("INSERT INTO embeddings (embedding, encoded_raw_payload, anchor_url, embedding_id, content_type) VALUES ($1, $2, $3, $4, $5)", embedding_data_rows)
    return


async def search_nearest_embeddings(db: Database, search_query_embeddings: numpy.ndarray, url: str):
    """
    Searches for the nearest embeddings within a particular document
    """
    anchor_url = await base64_encode_string(url)
    search_results = await db.fetch_all("SELECT embedding_id, encoded_raw_payload FROM embeddings WHERE anchor_url = $1 AND content_type = 'd' ORDER by embedding <-> $2 LIMIT 10", anchor_url, search_query_embeddings)
    return search_results
