""" Script to run wiki search using cohere api
"""
# Import necessary libraries
from datasets import load_dataset
import torch
import cohere
import numpy as np
from config import WIKI_EMB, TOP_K, MAX_DOCS

def load_data(max_docs: int, wiki_emb: str) -> (dict, np.array):
    """ Load the specified amount of data through streaming.

    Args:
        max_docs (int): maximum documents to load.
        wiki_emb (str): dataset name.

    Returns:
        (dict, np.array): document dictionary and embeddings.
    """
    # Load at max documents + embeddings
    docs_stream = load_dataset(wiki_emb, split="train", streaming=True)
    
    # Store the document content and the embeddings
    docs = []
    doc_embeddings = []
    for doc in docs_stream:
        docs.append(doc)
        doc_embeddings.append(doc['emb'])
        if len(docs) >= max_docs:
            break

    doc_embeddings = torch.tensor(doc_embeddings)
    return (docs, doc_embeddings)


def get_client(token: str) -> object:
    """ Get the cohere API client.

    Args:
        token (str): API key.

    Returns:
        object: client.
    """
    co = cohere.Client(token)  
    return co

def get_query_embeddings(query: list, model:str, token: str) -> np.array:
    """ Get embeddigns for the query

    Args:
        query (list): list of queries.
        model (str): model name.
        token (str): cohere API key.

    Returns:
        np.array: query embeddings.
    """
    co = get_client(token)  
    response = co.embed(texts=query, model=model)
    query_embedding = response.embeddings 
    query_embedding = torch.tensor(query_embedding)
    return query_embedding

def get_top_k_cohere(query: str, docs: dict, 
            q_emb: np.array, d_emb: np.array, 
            top_k: int) -> (list, list):
    """ Get top k documents based on similarity score.

    Args:
        query (str): user query.
        docs (dict): document dictionary.
        q_emb (np.array): query embedding.
        d_emb (np.array): document embedding
        top_k (int): top k documents.

    Returns:
        (list, list): list of title, list of text.

    """
    # Compute dot score between query embedding and document embeddings
    dot_scores = torch.mm(q_emb, d_emb.transpose(0, 1))
    top_k = torch.topk(dot_scores, k=top_k)

    # Print results
    titles = []
    texts = []
    print("Query:", query)
    for doc_id in top_k.indices[0].tolist():
        titles.append(docs[doc_id]['title'])
        texts.append(docs[doc_id]['text'])
        print(docs[doc_id]['title'])
        print(docs[doc_id]['text'], "\n")
    return (titles, texts)


def main(query: str, model: str, token: str):
    """ Method to call the workflow.

    Args:
        query (str): user query.
        model (str): model name.
        token (str): cohere API toekn.
    """
    # Add your cohere API key from www.cohere.com
    query_embedding = get_query_embeddings(query, model, token)

    # Load data
    docs, doc_embeddings = load_data(max_docs=MAX_DOCS, wiki_emb=WIKI_EMB)

    # Get top k documents matching the query
    title, text = get_top_k_cohere(query, docs, query_embedding, doc_embeddings, TOP_K)
    print(title, text)


