import os

import weaviate
from langchain.vectorstores import Weaviate


def create_class(
    client: weaviate.Client, drop: bool = False, class_name: str = "Document"
):
    if drop:
        client.schema.delete_class(class_name)
    schema = {
        "class": class_name,
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"vectorizeClassName": "false"}},
        "properties": [
            {"name": "title", "dataType": ["text"]},
            {"name": "content", "dataType": ["text"]},
        ],
    }
    if not client.schema.exists(class_name):
        client.schema.create_class(schema)


def get_store():
    weaviate_client = weaviate.Client(
        f'http://{os.environ["WEAVIATE_SERVICE_NAME"]}:{os.environ["WEAVIATE_PORT"]}'
    )

    create_class(
        weaviate_client,
        os.environ.get("WEAVIATE_DROP_COLLECTION", "False") == "True",
        os.environ.get("WEAVIATE_COLLECTION", "Document"),
    )

    vectorstore = Weaviate(
        client=weaviate_client,
        index_name=os.environ.get("WEAVIATE_COLLECTION", "Document"),
        text_key="content",
    )

    yield vectorstore
