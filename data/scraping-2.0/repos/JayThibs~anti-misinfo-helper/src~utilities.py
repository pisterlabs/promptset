import numpy as np
import openai


def get_text_embedding(text: str) -> np.ndarray:
    """
    Get the semantic embedding for a given text using OpenAI's model.

    Args:
    text (str): The text to get the embedding for.

    Returns:
    np.ndarray: The embedding vector for the given text.
    """
    response = openai.Embedding.create(
        model="text-similarity-babbage-001", input=[text]
    )
    return np.array(response["data"][0]["embedding"])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between vec1 and vec2.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
