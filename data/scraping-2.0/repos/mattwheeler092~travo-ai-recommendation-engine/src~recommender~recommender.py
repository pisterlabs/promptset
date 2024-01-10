import pinecone
import openai
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Dict
from .itinerary import Itinerary
from .utils import generate_title

from config import (
    PINECONE_API_KEY, 
    PINECONE_ENV, 
    PINECONE_INDEX,
    PINECONE_TOP_K,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL
)

def generate_api_response(
        destination: str,
        namespace: str,
        num_days: int,
        user_query: str,
        user_preferences: List[str]) -> Dict:
    """ Generates a recommendation API response based on user 
        query and preferences.
    Args:
        destination: A string representing the user's intended travel 
            destination.
        namespace: A string representing the namespace of the Pinecone 
            database to query.
        num_days: An integer representing the number of days the user 
            intends to travel for.
        user_query: A string representing the user's query.
        user_preferences: A list of strings representing the user's 
            preferences.
    Returns:
        A dictionary representing the recommendation API response 
            with a title and itinerary.
    """
    text_prompt = f"{user_query}. {', '.join(user_preferences)}"
    embedding = get_text_embedding(text_prompt)

    events = query_pinecone_db(embedding, namespace)
    itinerary = Itinerary(events, embedding)
    
    return {
        "title": generate_title(destination, num_days),
        "itinerary": itinerary.generate_itinerary(num_days)
    }


def query_pinecone_db(
        embedding: List[float], namespace: str) -> List[Dict]:
    """ Query Pinecone database for the nearest neighbors 
        of the given embedding.
    Args:
        embedding (List[float]): The vector embedding to query for.
        namespace (str): The namespace in which to query for the embedding.
    Returns:
        List[Dict]: A list of dictionaries representing the top 
            matching vectors and their metadata.
    """
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(PINECONE_INDEX)
    query_response = index.query(
        namespace=namespace,
        top_k=PINECONE_TOP_K,
        include_values=True,
        include_metadata=True,
        vector=embedding
    )
    return query_response["matches"]


def get_text_embedding(text: str) -> List[float]:
    """ Get a vector embedding for the given text using OpenAI's API.
    Args:
        text (str): The text to get the embedding for.
    Returns:
        List[float]: The vector embedding for the given text.
    """
    openai.api_key = OPENAI_API_KEY
    response = openai.Embedding.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL
    )
    return response["data"][0]["embedding"]