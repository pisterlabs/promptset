"""Collection of utilities to interact with or extend of standard model capabilities
"""

import tiktoken
import openai
import time
import lib.params as prm
import numpy as np
# import logging

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from transformers import logging as hf_logging

# to drop warnings from tokenizers
hf_logging.set_verbosity_error()


def get_tokens(message, method='SBERT'):
    """Count number of tokens in a message
    """

    if method == 'openai':
        model == prm.OPENAI_MODEL
        encoder = tiktoken.encoding_for_model(model)


    elif method == 'SBERT':
        model = SentenceTransformer(prm.SENTENCE_TRANSFORMER_MODEL)
        encoder = model.tokenizer

    return encoder.encode(message)


def decode_tokens(tokens, method='SBERT'):
    """Decode tokens
    """

    if method == 'openai':
        model == prm.OPENAI_MODEL
        encoder = tiktoken.encoding_for_model(model)

    elif method == 'SBERT':
        model = SentenceTransformer(prm.SENTENCE_TRANSFORMER_MODEL)
        encoder = model.tokenizer

    return encoder.decode(tokens)


def get_embedding_gpt(
        text, 
        model=prm.OPENAI_MODEL_EMBEDDING,
        calls_per_minute=60
        ):
    """Returns the embedding for a given text. Limit calls per minute
    Makes use of Open AI API to embed the text.
    """
    time_delay = 60 / calls_per_minute
    
    embedding = openai.Embedding.create(
        input=text, 
        model=model,
        )['data'][0]['embedding']
    
    time.sleep(time_delay)
    return embedding


def get_embedding_sbert(
        text,
        model=prm.SENTENCE_TRANSFORMER_MODEL
        ):
    """Returns the embedding for a given text.
    Makes use of Sentence-BERT (SBERT) to embed the text.
    """

    model = SentenceTransformer(model)
    embedding = model.encode(text)
    
    return embedding


def vector_similarity(x, y):
    """
    Returns a dot product of two vectors. For embedding values between 0 and 1 it is an equivalent
    of cosine similarity.
    """

    # Catch all ys that are not lists or arrays
    if not isinstance(y, (list, np.ndarray)):
        return 0

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)


    if len(x) < len(y):
        # pad x with zeros to match the length of y
        x = np.concatenate([x, np.zeros(y.shape[0] - x.shape[0], dtype=np.float32)])
    elif len(y) < len(x):
        # pad y with zeros to match the length of x
        y = np.concatenate([y, np.zeros(x.shape[0] - y.shape[0], dtype=np.float32)])

    return np.dot(x,y)


def order_document_sections_by_query_similarity(query, contexts):
    """
    Get embedding for the supplied query, and compare it against all of the available document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    
    # TODO: probable inefficiency - this embedding is later calculated
    # again downstream when saving to the database.
    query_embedding = get_embedding_sbert(query)
    
    document_similarities = sorted(
        [
            (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
            ], reverse=True
        )
    
    return document_similarities