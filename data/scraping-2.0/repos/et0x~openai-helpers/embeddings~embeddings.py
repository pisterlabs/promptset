from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from text.chunk import chunked_tokens

import numpy as np
import openai

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    """
    Get the embedding of a given text or tokens using OpenAI's Embedding API.

    Parameters:
    text_or_tokens (str or list): The text or tokens to be embedded.
    model (str): The name of the model to use for embedding. Defaults to the EMBEDDING_MODEL global variable.

    Returns:
    list: A list of floats representing the embedding of the input text or tokens.

    This function uses the OpenAI Embedding API to create an embedding for the given text or tokens using the specified model. It returns a list of floats representing the embedding.
    """
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]

def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    """
    This function generates embeddings for a text in a safe way. It will chunk the text into smaller chunks if it is too long.
    
    Parameters:
    text: string
        The input text to generate the embeddings.
    model: obj, optional
        Embedding model. Default is EMBEDDING_MODEL.
    max_tokens: int, optional
        Maximum number of tokens per chunk. Default is EMBEDDING_CTX_LENGTH.
    encoding_name: string, optional
        The name of the encoding type to use. Default is EMBEDDING_ENCODING.
    average: bool, optional
        Whether to return the average of embeddings or not. Default is True.
        
    Returns:
    list of floats:
        The embeddings of the input text.
    """
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings