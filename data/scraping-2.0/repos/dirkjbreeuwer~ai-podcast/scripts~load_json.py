from langchain.document_loaders import JSONLoader

import json
from pathlib import Path
from pprint import pprint

file_path='./data/raw/articles.json'


# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["title"] = record.get("title")
    metadata["author"] = ", ".join(record.get("author", []))  # Joining authors if it's a list

    return metadata


loader = JSONLoader(
    file_path=file_path,
    jq_schema='.[].text',
    metadata_func = metadata_func)

data = loader.load()

pprint(data)