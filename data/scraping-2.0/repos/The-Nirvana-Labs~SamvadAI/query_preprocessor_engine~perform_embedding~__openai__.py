import openai
import os
import torch


def perform_embedding(text):
    """
    Performs text embedding using the OpenAI API.

    Args:
    - text (str): The input text to perform embedding on.

    Returns:
    - torch.Tensor: The resulting text embedding as a tensor.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embedding = openai.Embedding("text-davinci-002")
    response = embedding.embed_text(text)
    return torch.tensor(response)
