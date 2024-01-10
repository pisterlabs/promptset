import openai
import numpy as np

def create_embedding(text: str, api_key: str = None) -> np.ndarray:
    """
    Creates an embedding for the given text using the OpenAI API.

    Args:
        text (str): The input text to create an embedding for.
        api_key (str, optional): The OpenAI API key. Defaults to None.

    Returns:
        np.ndarray: The embedding of the text as a NumPy array.

    """
    openai.api_key = api_key
    result = openai.Embed.create(model="text-embedding-ada-002", inputs=text)
    return result["data"][0]["embedding"]

def calc_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two embeddings.

    Args:
        a (np.ndarray): The first embedding as a NumPy array.
        b (np.ndarray): The second embedding as a NumPy array.

    Returns:
        float: The cosine similarity between the two embeddings.

    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)
