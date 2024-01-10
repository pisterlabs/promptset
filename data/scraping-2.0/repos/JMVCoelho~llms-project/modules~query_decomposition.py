import openai
import json
import re


# Implement multiple ways of decomposing queries
# Each one should assume the input as the base query dictionary, and output a json with the format:
# {<query_id>: {<decomposed_query_1>: text, ..., <decomposed_query_n>: text}, ...}

GPT_MODEL = "gpt-4"
openai.api_key = ""
pattern = re.compile(r"\d\.\s(.*)")
prompt = """
Original Query: {}
\nDecomposed Query: 
"""


def generate(content):
    """
    """
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system',
             'content': 'You are a utility that decomposes a complex query into smaller sentences that are more relevant to the movie using the examples given.'},
            {'role': 'user', 'content': content},
        ],
        model='gpt-4',
        temperature=0)
    return response['choices'][0]['message']['content']


def llm_based_decomposition(dataset, outpath):
    decomposed_queries = {}
    for q in dataset.queries_iter():
        subqueries = {}
        decomposed = generate(prompt.format(q.text))
        for i, sentence in enumerate(decomposed.split("\n")):
            try:
                sentence = pattern.match(sentence).groups()[0]
            except Exception as e:
                pass
            subqueries[i+1] = f"{q.title}. {sentence}"

        decomposed_queries[q.query_id] = subqueries
    queries_file = f"{outpath}/llm_decomposed_queries.json"

    with open(queries_file, 'w', encoding="utf-8") as fp:
        json.dump(decomposed_queries, fp)

    return queries_file


def sentence_decomposition(dataset, outpath):
    """Baseline decomposition method.
        Each query is decomposed into its sentences as provided in the data.
        Title of orignal query prepended to each decomposed subquery.
    """
    decomposed_queries = {}
    for q in dataset.queries_iter():
        subqueries = {}
        for sentence in q.sentence_annotations:
            subqueries[sentence['id']] = f"{q.title}. {sentence['text']}"
        
        decomposed_queries[q.query_id] = subqueries
    
    queries_file = f"{outpath}/sentence_decomposed_queries.json"

    with open(queries_file, 'w', encoding="utf-8") as fp:
        json.dump(decomposed_queries, fp)

    return queries_file

