import os
import sys
import pandas as pd
import random
import json
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ('neo4j', 'ioana123')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Neo4JConnector.NeoAlgorithms import (NeoAlgorithms)
from Neo4JConnector.NeoConnector import (NeoConnector)

openai.api_key = os.environ.get('OPENAI_API_KEY')


def create_file(dataset, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    print(f"Creating file: {file_name} for {dataset.shape} elements")
    for index, row in dataset.iterrows():
        query = "Write a short debunk for: " + row['statement']
        dic = {
            "messages": [
                {"role": "system", "content": "You are a fact-checking journalist"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": row["summary_explanation"]},
            ]
        }
        with open(file_name, 'a+') as json_file:
            json_file.write(json.dumps(dic) + "\n")


def prepare_data():
    neo_algo = NeoAlgorithms()
    neo_connector = NeoConnector()

    neo_algo.run_louvain()
    communities = neo_connector.get_popular_statements_communities()

    all_statements_not_top = []
    top = []
    all_statements = []
    for community in communities:
        statements = community['statements']
        if len(statements) > 2:
            top.extend(statements[:2])
            all_statements_not_top.extend(statements[2:])
        else:
            all_statements_not_top.extend(statements)
        all_statements.extend(statements)
    print(f"All statements processed: {len(all_statements)}")

    test_size = int(len(all_statements_not_top) * 0.15)
    test_batch = random.sample(all_statements_not_top, test_size)

    print(f"Test size: {test_size}")

    all_statements = [x for x in all_statements if x not in test_batch]
    all_statements_not_top = [x for x in all_statements_not_top if x not in test_batch]

    print(f"Lenght of top elements: {len(top)}")

    random_batch = random.sample(all_statements_not_top, len(top))
    print(f"Lenght of random batch: {len(random_batch)}")
    print(f"Lenght of full batch elements: {len(all_statements)}")

    df = pd.read_csv('../../data/data.csv', index_col=None)
    # test_batch

    test_batch = df[df['id'].isin(test_batch)]
    test_batch = test_batch[['id', 'statement', 'summary_explanation']]
    test_batch.to_csv("test.csv")

    # create train file with top 2
    top_batch = df[df['id'].isin(top)]
    create_file(top_batch, 'data/top_train.json')

    # create train file with random
    random_batch = df[df['id'].isin(random_batch)]
    create_file(random_batch, 'data/random_train.json')

    # create train with all
    all_batch = df[df['id'].isin(all_statements)]
    create_file(all_batch, 'data/all_train.json')


def call_train_models(train_files):
    for train_file in train_files:
        response = openai.File.create(
            file=open(train_file, "rb"),
            purpose='fine-tune'
        )
        with open("data/logs_models_names2.txt", 'a+') as file:
            file.write(train_file + "\n" + str(response) + "\n")
            print(response)
            response = openai.FineTuningJob.create(training_file=response['id'], model="gpt-3.5-turbo")
            print(response)
            file.write("\n" + str(response) + "\n")


def query_chat(model, text):
    query = "Write a short debunk for: " + text
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a fact-checking journalist."},
            {"role": "user",
             "content": query},
        ]
    )
    return response['choices'][0]["message"]["content"]


def generate(model_list):
    df = pd.read_csv('data/test.csv', index_col=None)
    result_df = pd.DataFrame()
    for index, row in df.iterrows():
        statement = row['statement']
        print(f"{index} --  {statement}")
        for model in model_list:
            generated_debunk = query_chat(model["id"], statement)
            row[model["name"]] = generated_debunk
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
    result_df.to_csv("results.csv")


def evaluate(model_list):
    df = pd.read_csv('data/results.csv', index_col=None)
    all_similarities = []
    for index, row in df.iterrows():
        explanation = row['summary_explanation']
        texts = [explanation]
        for model in model_list:
            texts.append(row[model['name']])
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_similarities = cosine_similarity(tfidf_matrix)
        all_similarities.append(cosine_similarities[0])

    i = 0
    for model in model_list:
        cos = [x[i + 1] for x in all_similarities]
        print(f"The average for {model['name']} is: {str(sum(cos) / len(cos))}")
        print(cos)
        i = i + 1


if __name__ == "__main__":
    print(openai.FineTuningJob.list(limit=10))
    # response = openai.FineTuningJob.create(training_file="file-c1X3xIE5W9rVMqi9yPjLhcCO", model="gpt-3.5-turbo")
    # response = openai.FineTuningJob.create(training_file="file-DqNCHKnPTyJKReEZl5FLY0rU", model="gpt-3.5-turbo")
    # print(response)

    '''prepare_data()
    call_train_models(['top_train.json', 'random_train.json']) #'top_train.json',
    generate([
        {"id": "ft:gpt-3.5-turbo-0613:personal::7tW4uzOE",
         "name": "top_statements"
         },
        {"id": "ft:gpt-3.5-turbo-0613:personal::7tWVNKV4",
         "name": "random_statements"
         },
        {"id": "gpt-3.5-turbo",
         "name": "no_training"
         }
    ])

    evaluate([
        {"id": "ft:gpt-3.5-turbo-0613:personal::7tW4uzOE",
         "name": "top_statements"
         },
        {"id": "ft:gpt-3.5-turbo-0613:personal::7tWVNKV4",
         "name": "random_statements"
         },
        {"id": "gpt-3.5-turbo",
         "name": "no_training"
         }
    ])'''

    # evaluate([
    # {"id":"file-H0wCSBmijFrW3q0wDYN1dR2g",
    #  "name": "top_statements"
    # },
    # {"id": "file-2dX6l41FwoQpvnpd4kgzJq9p",
    #  "name": "random_statements"
    # },
    # {"id": "gpt-3.5-turbo",
    #  "name": "no_training"
    # }
    # ])
