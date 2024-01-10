import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from uuid import uuid4


load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
COLLECTION_SIZE = os.getenv("COLLECTION_SIZE")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


class QdrantManager:
    """
    A class for managing collections in the Qdrant database.

    Args:
        collection_name (str): The name of the collection to manage.
        collection_size (int): The maximum number of documents in the collection.
        port (int): The port number for the Qdrant API.
        host (str): The hostname or IP address for the Qdrant server.
        api_key (str): The API key for authenticating with the Qdrant server.
        recreate_collection (bool): Whether to recreate the collection if it already exists.

    Attributes:
        client (qdrant_client.QdrantClient): The Qdrant client object for interacting with the API.
    """

    def __init__(
        self,
        collection_name=COLLECTION_NAME,
        collection_size: int = COLLECTION_SIZE,
        port: int = QDRANT_PORT,
        host=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
        recreate_collection: bool = False,
    ):
        self.collection_name = collection_name
        self.collection_size = collection_size
        self.host = host
        self.port = port
        self.api_key = api_key

        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        self.setup_collection(collection_size, recreate_collection)

    def setup_collection(self, collection_size: int, recreate_collection: bool):
        if recreate_collection:
            self.recreate_collection()

        try:
            collection_info = self.get_collection_info()
            current_collection_size = collection_info["vector_size"]

            if current_collection_size != int(collection_size):
                raise ValueError(
                    f"""
                    Existing collection {self.collection_name} has different collection size
                    To use the new collection configuration, you need to recreate the collection as it already exists with a different configuration.
                    use recreate_collection = True.
                    """
                )

        except Exception as e:
            self.recreate_collection()
            print(e)

    def recreate_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.collection_size, distance=models.Distance.COSINE
            ),
        )

    def get_collection_info(self):
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )

        return {
            "points_count": int(collection_info.points_count),
            "vectors_count": int(collection_info.vectors_count),
            "indexed_vectors_count": int(collection_info.indexed_vectors_count),
            "vector_size": int(collection_info.config.params.vectors.size),
        }

    def upsert_point(self, id, payload, embedding):
        response = self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    payload=payload,
                    vector=embedding,
                ),
            ],
        )

        return response

    def upsert_points(self, ids, payloads, embeddings):
        response = self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                payloads=payloads,
                vectors=embeddings,
            ),
        )

        return response

    def search_point(self, query_vector, limit):
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
        )

        return response

    def delete_collection(self):
        response = self.client.delete_collection(collection_name=self.collection_name)

        return response


# remove in commit


def get_embedding(prompt, model="text-embedding-ada-002"):
    prompt = prompt.replace("\n", " ")

    embedding = None
    try:
        embedding = openai.Embedding.create(input=[prompt], model=model)["data"][0][
            "embedding"
        ]

    except Exception as err:
        print(f"Sorry, There was a problem {err}")

    return embedding


def get_embeddings(prompts, model="text-embedding-ada-002"):
    prompts = [prompt.replace("\n", " ") for prompt in prompts]

    embeddings = None
    try:
        embeddings = openai.Embedding.create(input=prompts, model=model)["data"]

    except Exception as err:
        print(f"Sorry, There was a problem {err}")

    return [embedding["embedding"] for embedding in embeddings]


if __name__ == "__main__":
    import openai

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    manager = QdrantManager()

    collection_info = manager.get_collection_info()
    print(collection_info)

    # # upsert a point
    # record_id = uuid4().hex
    # record_payload = {"title": "Dreaming big", "chunk": "wow this is great"}
    # record_embedding = get_embedding(record_payload["chunk"])

    # response = manager.upsert_point(
    #     record_id,
    #     record_payload,
    #     record_embedding,
    # )

    # # upsert batch
    # record_ids = [uuid4().hex for x in range(3)]
    # record_payloads = [
    #     {
    #         "title": "War",
    #         "chunk": """
    #             War is an intense armed conflict[a] between states, governments, societies, or paramilitary groups such as mercenaries, insurgents, and militias. It is generally characterized by extreme violence, destruction, and mortality, using regular or irregular military forces. Warfare refers to the common activities and characteristics of types of war, or of wars in general.[2] Total war is warfare that is not restricted to purely legitimate military targets, and can result in massive civilian or other non-combatant suffering and casualties.
    #             While some war studies scholars consider war a universal and ancestral aspect of human nature,[3] others argue it is a result of specific socio-cultural, economic or ecological circumstances.[4]
    #         """,
    #     },
    #     {
    #         "title": "Car",
    #         "chunk": """
    #             A car or automobile is a motor vehicle with wheels. Most definitions of cars say that they run primarily on roads, seat one to eight people, have four wheels, and mainly transport people (rather than goods).
    #      """,
    #     },
    #     {
    #         "title": "Horse",
    #         "chunk": "The horse (Equus ferus caballus)[2][3] is a domesticated, one-toed, hoofed mammal. It belongs to the taxonomic family Equidae and is one of two extant subspecies of Equus ferus. The horse has evolved over the past 45 to 55 million years from a small multi-toed creature, Eohippus, into the large, single-toed animal of today. Humans began domesticating horses around 4000 BCE, and their domestication is believed to have been widespread by 3000 BCE. Horses in the subspecies caballus are domesticated, although some domesticated populations live in the wild as feral horses. These feral populations are not true wild horses, as this term is used to describe horses that have never been domesticated. There is an extensive, specialized vocabulary used to describe equine-related concepts, covering everything from anatomy to life stages, size, colors, markings, breeds, locomotion, and behavior.",
    #     },
    # ]
    # record_embeddings = get_embeddings(
    #     [record_payload["chunk"] for record_payload in record_payloads]
    # )

    # response = manager.upsert_points(
    #     record_ids,
    #     record_payloads,
    #     record_embeddings,
    # )

    # # getting collection_info
    # collection_info = manager.get_collection_info()
    # print(collection_info)

    # # search point
    # query_embedding = get_embedding("what are my rights?")

    # response = manager.search_point(query_embedding, 1)

    # print(response)

    # # delete all points
    # res = manager.delete_collection()
    # print(res)
