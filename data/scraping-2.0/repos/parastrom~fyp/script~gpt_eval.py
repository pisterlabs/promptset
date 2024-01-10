import json
import sqlite3

import openai
from typing import Optional
import re
from sqlparse.tokens import Keyword
from sqlparse import parse
from collections import Counter
import datasets
from src.settings import DATASETS_PATH, OPENAI_KEY
from src.util import evaluation

openai.api_key = OPENAI_KEY


def gen_example(question):
    # Call GPT-3.5 Turbo API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that translates English questions into SQL queries."},
            {"role": "user", "content": f'Translate the following question into a SQL query, return only the SQL query, nothing more: "{question}"'}
        ]
    )

    # Compare the generated SQL query with the ground truth
    generated_query = response.choices[0].message["content"].strip()
    ' '.join(generated_query.split('\n'))

    return generated_query


def query_compare(db, orig_query, gen_query):

    def execute_query(db, query):
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error executing query: {e}")
            results = None

        conn.close()
        return results

    orig_res = execute_query(db, orig_query)
    gen_res = execute_query(db, gen_query)
    return orig_res == gen_res


def compute_component_wise_accuracy(generated_query, components):
    parsed_query = parse(generated_query)[0]

    component_wise_accuracy = {}
    for component, count in components.items():
        component_wise_accuracy[component] = count_matching_keywords(parsed_query, component) / count

    return component_wise_accuracy


def count_matching_keywords(parsed_query, component):
    return len([token for token in parsed_query.tokens if token.ttype == Keyword and token.value.upper() == component.upper()])


def update_component_wise_accuracy(component_wise_accuracy, evaluation_result):
    for component, accuracy in evaluation_result.items():
        if component in component_wise_accuracy:
            component_wise_accuracy[component].append(accuracy)
        else:
            component_wise_accuracy[component] = [accuracy]


def extract_components(query):
    parsed_query = parse(query)[0]
    keywords = [token.value.upper() for token in parsed_query.tokens if token.ttype == Keyword]
    return Counter(keywords)


def main():
    spider_data_dict = datasets.load_dataset(path="../src/process/loaders/spider.py", cache_dir=DATASETS_PATH, split='validation')

    gen_queries = []
    db_paths = []
    orig_queries = []

    idx = 0

    for idx in range(10):
        sample = spider_data_dict[idx]
        question = sample["question"]
        query = sample["query"]
        db_id = sample["db_id"]
        db_path = sample["db_path"] + "/" + db_id + "/" + db_id + ".sqlite"

        gen_query = gen_example(question)

        gen_queries.append(gen_query)
        db_paths.append(db_path)
        orig_queries.append(query)

    final_res = evaluation.evaluate_queries(orig_queries, gen_queries, db_paths)

    # Dump the final results to a JSON file
    with open("gpt3_5_text_to_sql_evaluation_results.json", "w") as f:
        json.dump(final_res, f, indent=2)


if __name__ == "__main__":
    main()
