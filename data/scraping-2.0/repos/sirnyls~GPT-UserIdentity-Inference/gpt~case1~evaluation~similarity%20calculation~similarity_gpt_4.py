import openai
import os
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def gpt4_text_similarity(text1, text2, model="gpt-4"):
    """
    Measures the similarity between two texts using GPT-4.

    Parameters:
    text1 (str): First text for comparison.
    text2 (str): Second text for comparison.
    model (str): The GPT model to use.

    Returns:
    float: A similarity score between 0 (not similar) and 1 (very similar).
    """

    prompt = f"Rate the similarity between the following two texts on a scale from 0 (completely different) to 1 (identical):\n\nText 1: {text1}\n\nText 2: {text2}"
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "user",
        "content":  prompt
        }
    ],
    max_tokens=10
    )
    #print(response.choices[0].message.content)
    # Extracting the similarity score from the response
    try:
        last_message = response.choices[0].message.content
        similarity_score = float(last_message.strip())
    except (ValueError, KeyError, IndexError):
        similarity_score = None
    print(similarity_score)
    return similarity_score
