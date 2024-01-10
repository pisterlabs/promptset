from typing import List

import numpy as np
import openai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential

TWITTER_TOKEN_LIMIT = 280


def search_material(topics: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    It takes a query and a dataframe of search embeddings, and returns the top n most similar documents

    :param topics: the dataframe containing the search column
    :type topics: pd.DataFrame with column `embedding`
    :param query: the query string
    :type query: str
    :return: A dataframe with the top n results from the search query.
    """
    embedding = get_embedding(query, engine="text-search-davinci-query-001")

    topics["similarity"] = topics.embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )

    return topics


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-similarity-davinci-001") -> List[float]:
    """
    It takes a string of text and returns embeddings for the text

    :param text: The text to embed
    :type text: str
    :param engine: The name of the engine to use, defaults to text-similarity-davinci-001 (optional)
    :return: A list of floats.
    """
    # Replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


def cosine_similarity(a, b):
    """
    It takes two vectors, a and b, and returns the cosine of the angle between them

    :param a: the first vector
    :param b: the number of bits to use for the hash
    :return: The cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def respond_using_topic(
    text: str, topic: str, max_tokens: int = TWITTER_TOKEN_LIMIT, temperature: int = 0
) -> str:
    if "instruction" in text or "command" in text:
        return None

    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. Using only the information and facts provided in the excerpt below, "
        f"respond to this tweet in less than {max_tokens} characters. Provide action items and show hope:"
        f"\n###\nTweet:{text}"
        f"\n###\nExcerpt:{topic}\n###\n\nResponse:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


def respond_generic(text: str, max_tokens: int = TWITTER_TOKEN_LIMIT, temperature: int = 0) -> str:
    if "instruction" in text or "command" in text:
        return None

    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"You are a climate change educator. "
        f"Respond to this tweet in less than {max_tokens} characters by specifically addressing any "
        "false points with factual information. Add additional background."
        f"-\n######\n-Tweet:{text}"
        f"Response:",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


def respond_mention(text: str, max_tokens: int = TWITTER_TOKEN_LIMIT, temperature: int = 0) -> str:
    """Create response to a direct @ mention"""
    if "instruction" in text or "command" in text:
        return None

    is_activity = completion_with_backoff(
        model="text-davinci-002",
        prompt="Is the input an activity that someone can do? Answer YES or NO."
        f"-\n######\n-Input:{text}"
        f"Response:",
        temperature=0,
        max_tokens=3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )["choices"][0]["text"].strip()

    if is_activity.lower() == "yes":
        return completion_with_backoff(
            model="text-davinci-002",
            prompt="Provide a list of 3 easy action items that an ordinary citizen "
            "can take in their daily lives to reduce carbon emissions when performing this activity. "
            f"Respond in less than {max_tokens} characters."
            f"-\n######\n-Activity:{text}"
            f"Response:",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )["choices"][0]["text"].strip()
    else:
        return respond_generic(text, max_tokens, temperature)


def split_responses(text: str) -> List[str]:
    """Split response into list of responses satisfying the token limit.
    """
    if len(text) <= TWITTER_TOKEN_LIMIT:
        return [text]

    num = 0
    responses = [""]
    for sentence in text.split(". "):
        if sentence == "":
            continue
        if len(sentence) > TWITTER_TOKEN_LIMIT - 5:
            words = sentence.split()
            k = int(len(words)/2)
            phrase1 = " ".join(words[:k]) + f" ({num + 1})"
            phrase2 = " ".join(words[k:]) + ". "
            responses[num] += phrase1
            responses.append("")
            num += 1
            responses[num] += phrase2

        elif len(sentence) + len(responses[num]) <= TWITTER_TOKEN_LIMIT - 5:
            responses[num] += sentence
            responses[num] += ". "
        else:
            if responses[num][-2:] == ". ":
                responses[num] += f"({num + 1})"
            else:
                responses[num] += f". ({num + 1})"
            responses.append("")
            num += 1
            responses[num] += sentence
            responses[num] += ". "

    if responses[-1] == "" or responses[-1] == "\n":
        responses = responses[:-1]
        
    if responses[-1][-2:] == ". ":
        responses[-1] += f"({num + 1})"

    return [r.replace("..", ".") for r in responses]
