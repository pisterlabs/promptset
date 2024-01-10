from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import time
import openai
import numpy as np
from dotenv import dotenv_values

# this script can be used to search the existing question database to answer previously answered questions
# this will bring down the cost of running such a service by fetching answer to every query via OpenAI matching

EMBEDDING_MODEL = "text-embedding-ada-002"
config = dotenv_values("../.env")
openai.api_key = config['OPENAI_API_KEY']

# matching a new query with a set of existing question
def fuzzy_wuzzy_match(question, list_of_prev_questions):
    Str_A = 'Read the sentence - My name is Ali'
    Str_B = 'My name is Ali'
    ratio = fuzz.token_set_ratio(Str_A, Str_B)
    print(ratio)

    # get a list of matches ordered by score, default limit to 5
    print(process.extract(question, list_of_prev_questions))
    print(process.extractOne(question, list_of_prev_questions))

# just to compare the performance, we're seeing how OpenAI performs on the same data
def open_ai_match(question, list_of_prev_questions):
    question_embedding, question_tokens = get_embedding(question)
    prev_questions_embeddings, cost_incurred = compute_questions_embeddings(list_of_prev_questions)
    print(question_tokens * 0.0004 / 1000)
    print(cost_incurred)

    document_similarities = sorted([
        (vector_similarity(question_embedding, prev_questions_embeddings[prev_question]), prev_question) for prev_question in prev_questions_embeddings.keys()
    ], reverse=True)

    print(document_similarities)

def compute_questions_embeddings(questions):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    embedding_dict = {}
    total_tokens_used = 0
    for question in questions:
        embedding, tokens = get_embedding(question)
        embedding_dict[question] = embedding
        total_tokens_used = total_tokens_used + tokens
    cost_incurred = total_tokens_used * 0.0004 / 1000
    return embedding_dict, cost_incurred

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    time.sleep(7)
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"], result["usage"]["total_tokens"]

def main():
    query = 'what is klima token'
    choices = ['what\'s klima', 'what klima dao', 'what is the klima token', 'what is klima dao']
    fuzzy_wuzzy_match(query, choices) # getting the match fr
    open_ai_match(query, choices)

if __name__ == '__main__':
    main()