import os
import csv
from openai import OpenAI

import numpy as np
from pprint import pprint

openAIclient = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def load(filepath):
    faq_db = []
    with open(filepath, 'r', encoding='utf-8') as fp:
        cr = csv.reader(fp)
        next(cr)

        for row in cr:
            # row: id,question,answer
            text = "Question: " + row[1] + "\nAnswer: " + row[2] + "\n"
            vector = get_embedding(text)
            doc = {'id': row[0], 'vector': vector,
                   'question': row[1], 'answer': row[2]}
            faq_db.append(doc)

    return faq_db


def get_embedding(content, model='text-embedding-ada-002'):
    response = openAIclient.embeddings.create(input=content, model=model)
    vector = response.data[0].embedding
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search(vector, db):
    results = []

    for doc in db:
        score = similarity(vector, doc['vector'])
        results.append(
            {'id': doc['id'], 'score': score, 'question': doc['question'], 'answer': doc['answer']})

    results = sorted(results, key=lambda e: e['score'], reverse=True)

    return results


if __name__ == '__main__':
    faq_db = load('prompt-faq.csv')
    # print(faq_db)

    question = "ReAct가 뭔가요?"
    vector = get_embedding(question)
    # print(question, vector)

    result = search(vector, faq_db)
    pprint(result)
