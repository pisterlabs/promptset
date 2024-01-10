import os

import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import httpx

COLLECTION_NAME = "ai_devs_newsletter"
OPENAI_EMBEDDING_SIZE = 1536

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Get task
task = client.get_task("search")
ic(task.data)

# Get question and create its embedding
question = task.data["question"]
question_embedding = (
    openai.embeddings.create(
        input=question,
        model="text-embedding-ada-002",
    )
    .data[0]
    .embedding
)

# Extract url from task msg
url = task.data["msg"].split(" - ")[1]

# Get json from url
response = httpx.get(url)
data = response.json()

# Initialize Qdrant client
q_client = QdrantClient(path="db/qdrant/")

# Chek if collection already exists
try:
    collection_info = q_client.get_collection(COLLECTION_NAME)
except ValueError:
    # Create collection as it does not exist
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=OPENAI_EMBEDDING_SIZE, distance=Distance.COSINE, on_disk=True
        ),
    )

    # Fetch collection info again
    collection_info = q_client.get_collection(COLLECTION_NAME)


# Check if documents are already indexed
if collection_info.points_count == 0:
    ic("Indexing documents...")
    points = []
    # Get embeddings for each article
    for i, entry in enumerate(data):
        ic(f"Indexing document {i}...")
        vector = (
            openai.embeddings.create(
                input=entry["info"],
                model="text-embedding-ada-002",
            )
            .data[0]
            .embedding
        )

        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "url": entry["url"],
                    "title": entry["title"],
                    "date": entry["date"],
                },
            )
        )

    ic("Inserting documents into Qdrant...")
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    # Refresh task as the above operation takes some time
    # and the token most likely expired
    task = client.get_task("search")


# Now that we have indexed documents, we can search for the answer
ic("Searching for answer...")
search_results = q_client.search(
    collection_name=COLLECTION_NAME,
    query_vector=question_embedding,
    limit=1,
)

ic(search_results)

answer = search_results[0].payload["url"]

# Post an answer
response = client.post_answer(task, answer)
ic(response)
