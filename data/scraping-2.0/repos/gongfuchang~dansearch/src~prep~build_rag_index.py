"""
Reads vault dictionary, creates embeddings for each chunk, and creates a rag index.
"""
import pickle
from typing import List

import numpy as np
import torch.nn.functional as F

from src.logger import logger
from src.utils.model_util import get_model_tuple, get_device, average_pool
from src.prep.build_vault_dict import get_vault
import pinecone
import tiktoken

import openai



# initialize connection to pinecone (get API key at app.pinecone.io)
PINECONE_API_KEY = "1abe81c4-be22-4f43-83c8-e10a7c6905ff"
# find your environment next to the api key in pinecone console
PINECONE_ENV = "us-west4-gcp-free"

INDEX_NAME = 'chess-kb'

# Prepare pinecone client and tokenizer
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
pinecone.whoami()
pinecone_index = pinecone.Index(INDEX_NAME)

embed_model = "embd-ada2"

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def build_batch_embeddings(document_batch) :
    """Embed a batch of documents

    Args:
        document_batch: List of documents to embed
        tokenizer: Tokenizer to tokenize documents; should be compatible with model
        model: Model to embed documents

    Returns:
        List of document embeddings
    """
    batch = document_batch
    values_batch = openai.Embedding.create(
        input=[' '.join(doc[1]['chunk'].split()) for doc in batch],
        engine=embed_model
    )
    values_batch = tokenizer.encode_batch()
    encoded_batch = [(document_batch[index][0], val, document_batch[index][1]) for index, val in enumerate(values_batch)]
    pinecone_index.upsert(encoded_batch)


def assembly_embedding_index(vault: dict) -> dict[int, str]:
    """Build an index that maps document embedding row index to document chunk-id. 
    Used to retrieve document id after ANN on document embeddings.

    Args:
        vault: Dictionary of vault documents

    Returns:
        Mapping of document embedding row index to document chunk-id
    """
    embedding_index = dict()
    embedding_idx = 0

    for chunk_id, doc in vault.items():
        if doc['type'] == 'doc':
            continue  # Skip embedding full docs as they are too long for semantic search and take a long time

        embedding_index[embedding_idx] = chunk_id
        embedding_idx += 1

    return embedding_index


def build_embedding(vault: dict, batch_size=200):
    """Embedding all document chunks and return embedding array

    Args:
        vault: Dictionary of vault documents
        batch_size: Size of document batch to embed each time. Defaults to 4.
    """
    docs_embedded = 0
    chunk_batch = []
    chunks_batched = 0
    embedding_list = []

    for chunk_id, chunk in vault.items():
        if chunk['type'] == 'doc':
            continue  # Skip embedding full docs as they are too long for semantic search and take a long time

        # Get path and chunks
        if docs_embedded % 100 == 0:
            logger.info(f'Embedding document: {chunk_id} ({docs_embedded:,})')
        docs_embedded += 1
        # chunk = ' '.join(doc['chunk'].split()  # Remove extra whitespace and add prefix

        # logger.info(f'Chunk: {processed_chunk}')
        chunk_batch.append([chunk_id, chunk])  # Add chunk to batch
        chunks_batched += 1

        if chunks_batched % batch_size == 0:
            # Compute embeddings in batch and append to list of embeddings
            build_batch_embeddings(chunk_batch)

            # Reset batch
            chunks_batched = 0
            chunk_batch = []

    # Add any remaining chunks to batch
    if chunks_batched > 0:
        build_batch_embeddings(chunk_batch)


def query_rag(query, tokenizer, model, doc_embeddings_array, n_results=3):
    query_tokenized = tokenizer(f'query: {query}', max_length=512, padding=False, truncation=True, return_tensors='pt').to(get_device())
    outputs = model(**query_tokenized)
    query_embedding = average_pool(outputs.last_hidden_state, query_tokenized['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1).detach().cpu().numpy()

    cos_sims = np.dot(doc_embeddings_array, query_embedding.T)
    cos_sims = cos_sims.flatten()

    top_indices = np.argsort(cos_sims)[-n_results:][::-1]

    return top_indices


if __name__ == '__main__':
    # Load docs
    vault = get_vault()
    logger.info(f'Vault length: {len(vault):,}')

    # Build and save embedding index
    embedding_index = assembly_embedding_index(vault)
    logger.info(f'Embedding index length: {len(embedding_index):,}')
    build_embedding(vault)


