import ndjson
import json
import sys
import os

from tqdm import tqdm
import openai
from dataclasses import dataclass, field
from uuid import uuid4
from typing import Optional, Literal


def batch_loader(seq, size):
    """
    Iterator that takes in a list `seq` and returns
    chunks of size `size`
    """
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


def text_of_entry(x):
    return (
        "/-- " + x["doc_string"] + " -/" + "\n" + x["formal_statement"]
        if x["doc_string"]
        else x["formal_statement"]
    )


def main():
    READ_DIR = "../parse_docgen/docgen_export_with_formal_statement.jsonl"
    OUT_DIR = "./embeddings.jsonl"

    if os.path.isfile(OUT_DIR):
        raise AssertionError(f"{OUT_DIR} is already a file")

    print("loading docgen data...")
    with open(READ_DIR) as f:
        data = ndjson.load(f)

    print("creating embeddings")
    for batch in tqdm(batch_loader(data, 100)):
        texts = [text_of_entry(x) for x in batch]

        responses = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002",
        )

        log = []
        for entry, response in zip(batch, responses["data"]):
            to_log = {"name": entry["name"], "embedding": response["embedding"]}
            log.append(to_log)

        with open(OUT_DIR, "a+") as f:
            jsonstr = ndjson.dumps(log)
            f.write(jsonstr + "\n")


if __name__ == "__main__":
    main()
