import json
import os
from typing import List

import numpy as np
import openai
import pandas as pd
from IPython.display import display
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Makes a warning go away
os.environ["TOKENIZERS_PARALLELISM"] = "true"
openai.api_key = os.environ.get("OPENAI_API_KEY")


def compare_records_to_df(record_pairs: List[List[str]], models):
    """compare_records_to_df Generate a pd.DataFrame cosine similarity comparisons for a list of record pairs and models

    Parameters
    ----------
    record_pairs : List[List[str]]
        Pairs of records to compare
    models : Dict[str, SentenceTransformer]
        A pair of sentence transformers to compare
    """

    rows = []
    for name_one, name_two in record_pairs:
        scores = []
        for model_name in models.keys():
            model = models[model_name]

            embedding_one = model.encode(name_one)
            embedding_two = model.encode(name_two)
            score = 1.0 - cosine(embedding_one, embedding_two)

            scores.append(score)

        data_obj = (
            openai.Embedding.create(input=[name_one, name_two], model="text-embedding-ada-002")
        )["data"]

        openai_embeddings = [d["embedding"] for d in data_obj]
        openai_score = 1.0 - cosine(openai_embeddings[0], openai_embeddings[1])

        rows.append([name_one, name_two, scores[0], scores[1], openai_score])

    df = pd.DataFrame(
        rows, columns=["Name One", "Name Two", "All Cosine", "Paraphrase Cosine", "OpenAI Cosine"]
    )

    return df


models = {
    "paraphrase-multilingual-MiniLM-L12-v2": SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "sentence-transformers/all-MiniLM-L12-v2": SentenceTransformer(
        "sentence-transformers/all-MiniLM-L12-v2"
    ),
}

name_pairs = np.array(
    [
        ["Russell H Jurney", "Russell Jurney"],
        ["Russ H. Jurney", "Russell Jurney"],
        ["Russ H Jurney", "Russell Jurney"],
        ["Russ Howard Jurney", "Russell H Jurney"],
        ["Russell H. Jurney", "Russell Howard Jurney"],
        ["Russell H Jurney", "Russell Howard Jurney"],
        ["Alex Ratner", "Alexander Ratner"],
        ["ʿAlī ibn Abī Ṭālib", "عَلِيّ بْن أَبِي طَالِب"],
        ["Igor Berezovsky", "Игорь Березовский"],
        ["Oleg Konovalov", "Олег Коновалов"],
        ["Ben Lorica", "罗瑞卡"],
        ["Sam Smith", "Tom Jones"],
        ["Sam Smith", "Ron Smith"],
        ["Sam Smith", "Samuel Smith"],
    ]
)

json_pairs = np.array(
    [
        [
            json.dumps({"name": "Russell H Jurney", "birthday": "02/01/1980"}),
            json.dumps({"name": "Russell Jurney", "birthday": "02/01/1990"}),
        ],
        [
            json.dumps({"name": "Russ H. Jurney", "birthday": "02/01/1980"}),
            json.dumps({"name": "Russell Jurney", "birthday": "02/01/1991"}),
        ],
        [
            json.dumps({"name": "Russ H Jurney", "birthday": "02/01/1980"}),
            json.dumps({"name": "Russell Jurney", "birthday": "02/02/1990"}),
        ],
        [
            json.dumps({"name": "Russ Howard Jurney", "birthday": "02/01/1980"}),
            json.dumps({"name": "Russell H Jurney", "birthday": "02/01/1990"}),
        ],
        [
            json.dumps({"name": "Russell H. Jurney", "birthday": "02/01/1980"}),
            json.dumps({"name": "Russell Howard Jurney", "birthday": "02/01/1990"}),
        ],
        [
            json.dumps({"name": "Russell H Jurney", "birthday": "02/01/1980"}),
            json.dumps({"name": "Russell Howard Jurney", "birthday": "02/01/1990"}),
        ],
        [
            json.dumps({"name": "Alex Ratner", "birthday": "02/01/1901"}),
            json.dumps({"name": "Alexander Ratner", "birthday": "02/01/1976"}),
        ],
        [
            json.dumps({"name": "ʿAlī ibn Abī Ṭālib", "birthday": "02/01/1980"}),
            json.dumps({"name": "عَلِيّ بْن أَبِي طَالِب", "birthday": "02/01/1980"}),
        ],
        [
            json.dumps({"name": "Igor Berezovsky", "birthday": "01/01/1980"}),
            json.dumps({"name": "Игорь Березовский", "birthday": "02/03/1908"}),
        ],
        [
            json.dumps({"name": "Oleg Konovalov", "birthday": "02/01/1980"}),
            json.dumps({"name": "Олег Коновалов", "birthday": "05/04/1980"}),
        ],
        [
            json.dumps({"name": "Ben Lorica", "birthday": "02/01/1980"}),
            json.dumps({"name": "罗瑞卡", "birthday": "02/01/1980"}),
        ],
        [
            json.dumps({"name": "Sam Smith", "birthday": "02/01/1980"}),
            json.dumps({"name": "Tom Jones", "birthday": "02/01/1976"}),
        ],
        [
            json.dumps({"name": "Sam Smith", "birthday": "02/01/1980"}),
            json.dumps({"name": "Ron Smith", "birthday": "02/01/2001"}),
        ],
        [
            json.dumps({"name": "Sam Smith", "birthday": "02/01/1980"}),
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/1801"}),
        ],
        [
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/1980"}),
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/1980"}),
        ],
        [
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/1980"}),
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/1991"}),
        ],
        [
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/1980"}),
            json.dumps({"name": "Samuel Smith", "birthday": "02/01/2011"}),
        ],
    ]
)

display(compare_records_to_df(name_pairs, models))

print()

display(compare_records_to_df(json_pairs, models))

print()
