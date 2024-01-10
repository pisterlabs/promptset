import pandas as pd
import openai
from scipy import spatial
import tiktoken
import ast  # for converting embeddings saved as strings back to arrays
from typing import List, Tuple
from config import GPT_MODEL, EMBEDDINGS_MODEL

def fetch_topic_embedding():
    topic_embeddings_df = pd.read_csv('data/embedding/topic_embedding.csv')
    # convert embeddings from CSV str type back to list type
    topic_embeddings_df['embedding'] = topic_embeddings_df['embedding'].apply(ast.literal_eval)

    return topic_embeddings_df

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> Tuple[List[str], List[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDINGS_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = '''Use the below articles on business process management, large_language_models, Natural Language Processing,
    Optical Character Recognition, Speech Recognition to answer the subsequent question. If the answer cannot be found in the articles,
    write "I could not find an answer." and write why you could't find an answer when possible'''
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message+question