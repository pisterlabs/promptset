import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from scipy import spatial  # for calculating vector similarities for search
from dotenv import load_dotenv
import os
load_dotenv()

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
FT_MODEL = os.getenv('FT_MODEL')
openai.api_key = os.getenv("OPENAI_API_KEY")

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 2
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    
    try:
        query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
        )
    except Exception as e:
        print(e)
        return ""
    
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["context"], relatedness_fn(query_embedding, row["embeddings"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    question = f"\nQuestion: {query}"
    context = ""
    for related_context in strings:
        context+= f"{related_context}\n"
    return context + question

def ask(
    question: str,
    filepath: str,
    model: str = FT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = True,
) -> str:
    df = pd.read_csv(filepath)
    df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
 
    message = query_message(question, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions using the provided context only"},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message