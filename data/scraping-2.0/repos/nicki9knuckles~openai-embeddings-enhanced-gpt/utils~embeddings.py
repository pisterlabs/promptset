import openai
from openai.embeddings_utils import cosine_similarity
import os
import pickle
from tenacity import retry, wait_random_exponential, stop_after_attempt
import itertools
import numpy as np

from utils.prompt import num_tokens_from_messages, get_messages


# Define a function to retrieve an embedding from the cache if present, otherwise request via OpenAI API
def embedding_from_string(
    embedding_cache_path, string, embedding_cache, model="text-embedding-ada-002"
):
    if (string, model) not in embedding_cache.keys():
        # Fetch the embedding
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")

        # # Create a temporary file and write the updated cache to it
        temp_file_path = embedding_cache_path + ".tmp"
        with open(temp_file_path, "wb") as temp_embedding_cache_file:
            pickle.dump(embedding_cache, temp_embedding_cache_file)

        # If writing was successful, replace the old cache with the new one
        os.rename(temp_file_path, embedding_cache_path)
    # else:
    #     print(f"USING CACHED EMBEDDING FOR {string[:20]}")

    try:
        return embedding_cache[(string, model)]
    except KeyError:
        print(f"KeyError for string--------------: {string[:5]}")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    if not isinstance(text, str):
        print(f"Warning: Invalid text {text} of type {type(text)}")
        return None  # or some other placeholder value

    # NOTE: remember to replace new line chars because of performance issues with OpenAI API
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]


def get_n_nearest_neighbors(query_embedding, embeddings, n: int):
    """
    :param query_embedding: The embedding to find the nearest neighbors for
    :param embeddings: A dictionary of embeddings, where the keys are the entity type (e.g. Movie, Segment)
        and the values are the that entity's embeddings
    :param n: The number of nearest neighbors to return
    :return: A list of tuples, where the first element is the entity and the second element is the cosine
        similarity between -1 and 1
    """

    # This is not optimized for rapid indexing, but it's good enough for this example
    # If you're using this in production, you should use a more efficient vector datastore such as
    # those mentioned specifically by OpenAI here
    #
    #  https://platform.openai.com/docs/guides/embeddings/how-can-i-retrieve-k-nearest-embedding-vectors-quickly
    #
    #  * Pinecone, a fully managed vector database
    #  * Weaviate, an open-source vector search engine
    #  * Redis as a vector database
    #  * Qdrant, a vector search engine
    #  * Milvus, a vector database built for scalable similarity search
    #  * Chroma, an open-source embeddings store
    #

    target_embedding = np.array(query_embedding)

    similarities = [
        (segment, cosine_similarity(target_embedding, np.array(embedding)))
        for segment, embedding in embeddings.items()
    ]

    # Sort by similarity and get the top n results
    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    return nearest_neighbors


def ask_embedding_store(
    MAX_PROMPT_SIZE, chat_model, enc, chat_history, embeddings, max_documents: int
) -> str:
    """
    Fetch necessary context from our embedding store, striving to fit the top max_documents
    into the context window (or fewer if the total token count exceeds the limit)

    :param question: The question to ask
    :param embeddings: A dictionary of Section objects to their corresponding embeddings
    :param max_documents: The maximum number of documents to use as context
    :return: GPT's response to the question given context provided in our embedding store
    """
    # Loop through messages in reverse to find the most recent user message
    for message in reversed(chat_history):
        if message["role"] == "user":
            question = message["content"]
            break
    query_embedding = get_embedding(question)

    nearest_neighbors = get_n_nearest_neighbors(
        query_embedding, embeddings, max_documents
    )
    messages: Optional[List[Dict[str, str]]] = None

    base_token_count = num_tokens_from_messages(
        get_messages([], question, []), chat_model
    )

    token_counts = [
        len(enc.encode(document.replace("\n", " ")))
        for document, _ in nearest_neighbors
    ]
    cumulative_token_counts = list(itertools.accumulate(token_counts))
    indices_within_limit = [
        True
        for x in cumulative_token_counts
        if x <= (MAX_PROMPT_SIZE - base_token_count)
    ]
    most_messages_we_can_fit = len(indices_within_limit)

    context = [x[0] for x in nearest_neighbors[: most_messages_we_can_fit + 1]]

    messages = get_messages(context, question, chat_history)

    #     print(f"Prompt: {messages[-1]['content']}")
    result = openai.ChatCompletion.create(model=chat_model, messages=messages)

    # print(f"Result----------------------: {result.choices[0].message['content']}")
    # return result.choices[0].message["content"]
    return result
