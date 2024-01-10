import json

from langchain.document_loaders import JSONLoader  # pip install jq
from langchain.embeddings.openai import OpenAIEmbeddings  # pip install tiktoken
from qdrant_client import QdrantClient  # pip install qdrant-client
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema.document import Document
from uuid import uuid4

import os
import jsonReader

# https://qdrant.tech/documentation/quick-start/
# This code initializes Qdrant collection ai_devs_search_task, and adds vectors from local documents to it
# then it searches the collection for the most similar vector to the query vector
# vectors for documents and the query are generated using OpenAI API ada / creating embeddings
# takes some time 3-5mins approx
if __name__ == '__main__':
    MEMORY_PATH = "search_task_vectordb_init.json"
    COLLECTION_NAME = "ai_devs_search_task"

    qdrant = QdrantClient("http://localhost:6333")  # url=os.environ['QDRANT_URL']) |qdrant = QdrantClient("localhost", port=6333)
    result = qdrant.get_collections()
    indexed = next((collection for collection in result.collections if collection.name == COLLECTION_NAME), None)
    print("Collections:", result)

    if not indexed:
        qdrant.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=1536, distance="Cosine", on_disk=True))
    collectionInfo = qdrant.get_collection(COLLECTION_NAME)

    if not collectionInfo.points_count:
        # loader = JSONLoader(MEMORY_PATH, '.content')  # not working for specific multiline jsons
        # memory = loader.load()[0]
        # with open(MEMORY_PATH, 'r', encoding="utf8") as handle:
        #     data = handle.read()
        #     persistent_memory = [json.loads(l) for l in data.splitlines('}')]

        persistent_memory = jsonReader.JsonReader().read(MEMORY_PATH)
        print("read", len(persistent_memory), "documents")
        documents = [Document(page_content=str(content),metadata={'source': COLLECTION_NAME, 'uuid': str(uuid4()), 'content': content}) for content in persistent_memory]

        points = []
        embeddings = OpenAIEmbeddings()
        for document in documents:
            embedding = embeddings.embed_documents([document.page_content])[0]
            points.append({
                'id': document.metadata['uuid'],
                'payload': document.metadata,
                'vector': embedding,
            })
        print("Created", len(points), "vectors")
        qdrant.upsert(COLLECTION_NAME, wait=True, points=points)
        print("Indexed", len(points), "documents")