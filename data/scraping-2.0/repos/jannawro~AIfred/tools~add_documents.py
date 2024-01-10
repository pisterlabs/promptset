from datetime import datetime
import os
import sys
import json
from uuid import uuid4
from langchain_community.embeddings.openai import OpenAIEmbeddings

from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient


"""
required environment variables:
QDRANT_URL
OPENAI_API_KEY

the scripts expects to be given filepaths to documents as args

document structure(should be a json file):
{
    "content": "...",
    "key": "..." // a memory key from memory schema
}
"""


def main():
    documents = []

    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    doc_store = Qdrant(
        client=client, collection_name="documents", embeddings=OpenAIEmbeddings()
    )
    date = datetime.now().strftime("%d/%m/%Y") + " (DD/MM/YYYY)"
    for arg in sys.argv[1:]:
        with open(arg) as json_data:
            data = json.load(json_data)
            documents.append(
                {
                    "content": data["content"],
                    "metadata": {
                        "key": data["key"],
                        "last_updated": date,
                        "uuid": str(uuid4()),
                    },
                }
            )

    doc_store.add_texts(
        texts=[document["content"] for document in documents],
        metadatas=[document["metadata"] for document in documents],
        ids=[document["metadata"]["uuid"] for document in documents],
    )


if __name__ == "__main__":
    main()
