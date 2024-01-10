import openai
import numpy as np
import tiktoken
import pandas as pd
import sys

from pathlib import Path, PurePath
sys.path.append(PurePath(Path(__file__).parents[1]).as_posix())
from src.utils.logging.custom_logging import logger

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    logger.debug(f"most relevant document sections {document_similarities}")
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, template: str, memory) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        section = SEPARATOR + document_section.content.replace("\n", " ")
        if chosen_sections_len > MAX_SECTION_LEN:
            logger.warning(f'SECTION LEN EXCEED TO MAX SECTION LEN({MAX_SECTION_LEN})')
            logger.warning(f'missed to include in prompt: {section}')
        else:
            logger.info(f'Context Proability is: {_}')
            logger.debug(f'Contex: {section}')
            chosen_sections.append(section)
            chosen_sections_indexes.append(str(section_index))
        #chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))

    #header = """You are TVS QA BOT. You are capable of answering questions reqarding TVS Owner Manual. Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    _prompt = template+ "\n" +memory.load_memory_variables({})['history'] + "\nQUESTION: " + \
              question +"\ncontent: " + "".join(chosen_sections) + "\nFINAL ANSWER:"
    return _prompt


def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        template: str,
        memory,
        show_prompt: bool = False,) -> str:
    """
    Facade function to get question from user and call model, eventually returns the answer to user

    :param query: question to docsgpt
    :param df: document in dataframe
    :param document_embeddings: embedding vector of document
    :param template: prompt
    :param show_prompt: to show prompt in stdout or not? boolean
    :return: answer from docgpt
    """
    prompt = construct_prompt(
            query,
            document_embeddings,
            df,
            template,
            memory
        )

    #prompt = prompt + "\n" + memory.load_memory_variables({})['history']
    if show_prompt:

        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )
    return response["choices"][0]["text"].strip(" \n")