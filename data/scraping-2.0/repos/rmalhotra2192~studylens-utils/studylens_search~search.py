import uuid
import openai
import requests
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.http.models import Batch as QdrantBatch
from .config import Config_OpenAI, Config_Qdrant


class OPEN_AI_API:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)

    def completion_request(self, book_info) -> [list, Any]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant designed to output JSON.",
                    },
                    {
                        "role": "user",
                        "content": self.generate_prompt(book_info, "book"),
                    },
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def embedding_request(
        self, resource_json=None, resource_type=None, for_query=False, query=None
    ) -> [list, Any]:
        embedding_model = "text-embedding-ada-002"

        if for_query:
            request_input = query
        else:
            request_input = self.generate_embedding_text(resource_json, resource_type)

        response = self.client.embeddings.create(
            input=request_input,
            model=embedding_model,
        )

        return response.data[0].embedding

    def generate_prompt(self, resource_json, resource_type) -> str:
        prompt = Config_OpenAI.prompts["data_enrichment"].format(resource_type)
        prompt += "\n\n" + resource_json
        return prompt

    def generate_embedding_text(self, resource_json, resource_type) -> str:
        prompt = ""
        for field_category in ["general_fields", resource_type + "_fields"]:
            for field in Config_OpenAI.embedding_field_types[field_category]:
                if field["field_type"] == "text":
                    if (
                        resource_json[field["field_name"]] != ""
                        and resource_json[field["field_name"]]
                    ):
                        prompt += (
                            "<FIELD_START>"
                            + field["field_name"]
                            + ": "
                            + str(resource_json[field["field_name"]])
                            + "<FIELD_END>"
                        )
                    else:
                        prompt += (
                            "<FIELD_START>"
                            + field["field_name"]
                            + ": "
                            + "N/A"
                            + "<FIELD_END>"
                        )
                elif field["field_type"] == "array":
                    if (
                        resource_json[field["field_name"]] != []
                        and resource_json[field["field_name"]]
                    ):
                        prompt += (
                            "<FIELD_START>"
                            + field["field_name"]
                            + ": "
                            + ", ".join(resource_json[field["field_name"]])
                            + "<FIELD_END>"
                        )
                    else:
                        prompt += (
                            "<FIELD_START>"
                            + field["field_name"]
                            + ": "
                            + "N/A"
                            + "<FIELD_END>"
                        )

        return prompt


class Qdrant_DB:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.client = QdrantClient(url=url, api_key=api_key)

    def create_collection(self, collection: str):
        collection_body = {
            "vectors": {
                "size": Config_Qdrant.vector_size,
                "distance": Distance.COSINE,
                "on_disk": True,
            }
        }

        qdrant_url = f"{self.url}/collections/{collection}"
        headers = {"Content-Type": "application/json"}
        response = requests.put(qdrant_url, json=collection_body, headers=headers)
        print(
            f"Collection:{Config_Qdrant.collection_name} | Action: Creation > Status: {response.content}"
        )

    def upload_resource(
        self,
        resources,
        resource_types,
        collection: str,
        batch_size: int = 100,
        open_ai_api: OPEN_AI_API = None,
    ) -> None:
        collections = self.client.get_collections()
        if collection not in collections:
            self.create_collection(collection)

        ids = []
        payloads = []
        vectors = []

        for idx, resource in enumerate(resources):
            payload = resource
            vector = open_ai_api.send_embeddings_request(resource, resource_types[idx])

            s_id = str(uuid.uuid4())
            ids.append(s_id)
            payloads.append(payload)
            vectors.append(vector)

        self.client.upsert(
            collection_name=collection,
            points=QdrantBatch(ids=ids, payloads=payloads, vectors=vectors),
        )

    def search(
        self,
        collection: str,
        query: str,
        open_ai_api: OPEN_AI_API = None,
        limit=10,
        offset=0,
        query_filter=None,
    ):
        return self.client.search(
            collection_name=collection,
            query_vector=open_ai_api.embedding_request(for_query=True, query=query),
            limit=limit,
            offset=offset,
            query_filter=query_filter,
        )
