import os
import csv
import openai
import numpy as np
from dotenv import load_dotenv
from redisvl.index import SearchIndex
from typing import List, Dict
import ast
import time
import argparse
from rich import print

# Load secrets
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(doc):
    response = openai.Embedding.create(
        input=doc, model=OPENAI_EMBEDDING_MODEL, encoding_format="float"
    )
    embedding = response["data"][0]["embedding"]
    return embedding


def get_data(filepath) -> List[Dict]:
    print(f"Using the input file {filepath} to generate embeddings...")
    records = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for d in reader:
            records.append(
                {
                    "Narration": d["Narration"],
                    "Category": d["Category"],
                    "Narration_Embedding": get_embedding(d["Narration"]),
                }
            )
        return records


def save_embeddings(filepath, dict_list):
    fieldnames = dict_list[0].keys()

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dict_list:
            writer.writerow(row)


def create_and_load_index(records):
    index = SearchIndex.from_yaml("index.yaml")
    index.connect("redis://localhost:6379")
    index.create(overwrite=True)
    index.load(records)


def get_data_from_file(filepath):
    print(f"Using the input file {filepath} to load embeddings...")
    records = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                continue
            embedding = ast.literal_eval(row[2])
            records.append(
                {
                    "Narration": row[0],
                    "Category": row[1],
                    "Narration_Embedding": np.array(
                        embedding, dtype=np.float32
                    ).tobytes(),
                }
            )
    return records


def main():
    # Command Line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-regen",
        action="store_true",
        help="Flush the target and reload all. Use very carefully, usually -a should suffice",
    )
    args = argparser.parse_args()

    if args.regen:
        records = get_data(filepath="data/labelled.csv")
        save_embeddings(
            filepath="data/embeddings.csv",
            dict_list=records,
        )
    start_load_redis = time.time()
    records_with_embeddings = get_data_from_file(filepath="data/embeddings.csv")
    create_and_load_index(records_with_embeddings)
    print(
        f"Vector Database Loaded! ( {round(time.time() - start_load_redis,2)} seconds )"
    )


if __name__ == "__main__":
    main()
