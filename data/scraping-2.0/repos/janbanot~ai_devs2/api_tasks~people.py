import os
import requests  # type: ignore
from openai import OpenAI
import uuid
from dotenv import load_dotenv
from ai_devs_task import Task
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.embeddings import OpenAIEmbeddings
from typing import Dict, Any

SOURCE_URL = "https://zadania.aidevs.pl/data/people.json"

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

people: Task = Task(ai_devs_api_key, "people")
token: str = people.auth()
task_content: Dict[str, Any] = people.get_content(token)

qdrant = QdrantClient("localhost", port=6333)
embeddings = OpenAIEmbeddings()
COLLECTION_NAME = "people_task"

collections = qdrant.get_collections()
collection_names = [element.name for element in collections.collections]
if not (COLLECTION_NAME in collection_names):
    qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            on_disk_payload=True
        )

collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)

if collection_info.points_count == 0:
    people_data = requests.get(SOURCE_URL).json()
    for element in people_data:
        name = f"{element['imie']} {element['nazwisko']}"
        element_copy = element.copy()
        if "imie" in element_copy:
            del element_copy["imie"]
        if "nazwisko" in element_copy:
            del element_copy["nazwisko"]
        metadata = {
            "source": COLLECTION_NAME,
            "content": element_copy,
            "id": uuid.uuid4().hex,
        }
        point = name
        point_id = metadata["id"]
        point_vector = embeddings.embed_query(point)
        point_struct = {"id": point_id, "payload": metadata, "vector": point_vector}
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[point_struct]
        )

prompt_name: str = """
Extract the name and surname of the person from the text below.
Answer with name and surname only.
Examples:
Ulubiony kolor Agnieszki Rozkaz, to?
expected: Agnieszka Rozkaz
"""
name_surname = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_name},
            {"role": "user", "content": task_content["question"]}
        ]
    )
name_surname_string = name_surname.choices[0].message.content or ""
query_embedding = embeddings.embed_query(name_surname_string)
search_result = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_embedding,
    limit=1,
    query_filter={
        "must": [
            {
                "key": "source",
                "match": {
                    "value": COLLECTION_NAME
                }
            }
        ]
    }
)
info = str(search_result[0].payload["content"])  # type: ignore
prompt_info: str = f"""
Answer the question about the person based on the text below
{info}
"""

api_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_info},
            {"role": "user", "content": task_content["question"]}
        ]
    )
answer = api_response.choices[0].message.content or ""
answer_payload: Dict[str, str] = {"answer": answer}
task_result: Dict[str, Any] = people.post_answer(token, answer_payload)
print(task_result)
