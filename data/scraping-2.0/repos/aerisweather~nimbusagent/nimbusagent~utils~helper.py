import logging
import os
from typing import Union, List, Iterable, Any

from numpy import dot
from numpy.linalg import norm
from openai import OpenAI

FUNCTIONS_EMBEDDING_MODEL = "text-embedding-ada-002"


def is_query_safe(query: str, api_key=None) -> bool:
    """Returns True if the query is considered safe, False otherwise.
    :param query: The query to check.
    :param api_key: The OpenAI API key to use. Uses the OPENAI_API_KEY environment variable if not provided.
    :return: True if the query is considered safe, False otherwise.
    """
    client = OpenAI(api_key=api_key if api_key else os.environ["OPENAI_API_KEY"])

    try:
        response = client.moderations.create(input=query)

        if response and response.results:
            result = response.results[0]

            if result and result.flagged:
                logging.debug(f"Query '{query}' was flagged by OpenAI's moderation API. {result}")
                return False

        return True

    except Exception as e:
        logging.error(f"An error occurred while checking query safety: {e}")

    return False


def get_embedding(text, model=FUNCTIONS_EMBEDDING_MODEL, api_key=None):
    """Returns the embedding of the given text.
    :param text: The text to get the embedding of.
    :param model: The model to use. Defaults to the text-embedding-ada-002 model.
    :param api_key: The OpenAI API key to use. Uses the OPENAI_API_KEY environment variable if not provided.
    :return: The embedding of the given text.
    """
    try:
        text = text.replace("\n", " ")
        client = OpenAI(api_key=api_key if api_key else os.environ["OPENAI_API_KEY"])
        embedding = client.embeddings.create(
            input=text,
            model=model)
        return embedding.data[0].embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def cosine_similarity(list1, list2):
    """ get cosine similarity of two vector of same dimensions
    :param list1: The first vector.
    :param list2: The second vector.
    :return: The cosine similarity of the two vectors.
    """
    return dot(list1, list2) / (norm(list1) * norm(list2))


def find_similar_embedding_list(query: str, function_embeddings: list, k_nearest_neighbors: int = 1,
                                min_similarity: float = 0.5):
    """
    Return the k function descriptions most similar to given query.
    :param query: The query to check.
    :param function_embeddings: The list of function embeddings to compare to.
    :param k_nearest_neighbors: The number of nearest neighbors to return.
    :param min_similarity: The minimum cosine similarity to consider a function relevant.
    :return: The k function descriptions most similar to given query.
    """
    if not function_embeddings or len(function_embeddings) == 0 or not query:
        return None

    query_embedding = get_embedding(query)
    if not query_embedding:
        return None

    similarities = []
    for function_embedding in function_embeddings:
        similarity = cosine_similarity(query_embedding, function_embedding['embedding'])
        if similarity >= min_similarity:
            similarities.append({'name': function_embedding['name'], 'similarity': similarity})

    # Sort the results by similarity in descending order (most similar first)
    sorted_similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

    # Return the top k nearest neighbors
    return sorted_similarities[:k_nearest_neighbors]



def combine_lists_unique(list1: Iterable[Any], set2: Union[Iterable[Any], set]) -> List[Any]:
    """Combine two lists, removing duplicates.
    :param list1: The first list.
    :param set2: The second list. This can be a set or any iterable.
    :return: The combined list, with duplicates removed.
    """
    if isinstance(list1, list):
        new_list = list1.copy()
    else:
        new_list = list(list1)  # Convert iterable to list

    for item in set2:
        if item not in new_list:
            new_list.append(item)

    return new_list
