import openai
import pickle
import numpy as np
from typing import List
from openai.embeddings_utils import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

"""
Code search and relevance models:
	code-search-ada-code-001
	code-search-ada-text-001
	code-search-babbage-code-001
	code-search-babbage-text-001
"""


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-similarity-davinci-001") -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


def get_embedding_vector(snippet, engine="text-similarity-davinci-001"):
    snippet_embedding = get_embedding(snippet, engine)
    return np.array(snippet_embedding)


def snippet_similarity(snippet_1, snippet_2, engine):
    snippet_1_embedding = get_embedding(snippet_1, engine)
    snippet_2_embedding = get_embedding(snippet_2, engine)
    cos_similarity = cosine_similarity(snippet_1_embedding, snippet_2_embedding)
    return cos_similarity


def throw_literals(engine):
    code_snippet_1 = """
    # JavaScript
    if(spaceBefore !== NULL && localStorage.remainingSpace === spaceBefore) {
        throw 'QUOTA_EXCEEDED_ERR';
    }
"""

    code_snippet_2 = """
   # JavaScript 
   if (prev_output.is_stable === 0)
        throw \"prev is not stable\";
    if (prev_output.is_serial === 1 && prev_output.sequence !== 'good')
    """
    score = snippet_similarity(code_snippet_1, code_snippet_2, engine)
    print(f"similarity score:{score}")
    return score


def throw_literals_vs_for_in(engine):
    code_snippet_1 = """ 
   # JavaScript 
   for (var option in fieldData.options) {
        option = fieldData.options[option];
"""

    code_snippet_2 = """
   # JavaScript 
   if (prev_output.is_stable === 0)
        throw \"prev is not stable\";
    if (prev_output.is_serial === 1 && prev_output.sequence !== 'good')
    """
    score = snippet_similarity(code_snippet_1, code_snippet_2, engine)
    print(f"similarity score:{score}")
    return score


def main():
    engines = ['code-search-ada-code-001',
               'code-search-ada-text-001',
               'code-search-babbage-code-001',
               'code-search-babbage-text-001']

    for engine in engines:
        #print("both examples are throw literals")
        #throw_literals(engine)  #
        #print("=================================")
        #print("throw literals and for in comparison")
        #throw_literals_vs_for_in(engine)  # 0.74
        print("=================================")
        print(f"engine: {engine}, "
              f"throw_literals(engine)={throw_literals(engine)}, "
              f"throw_literals_vs_for_in(engine)={throw_literals_vs_for_in(engine)}")

    # ada - throw_literals: 0.79, throw_literals_vs_for_in: 0.74

if __name__ == "__main__":
    main()
