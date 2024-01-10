from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
from utils import CustomFormatter
import openai
import time
import os
from dotenv import load_dotenv, dotenv_values

# Env startup
load_dotenv()
_dotenv = dotenv_values(".env")

class EmbeddingsManager:
    def __init__(self,mode: str,
    path: str,
    collection_name: str,
    embedding_model: str=_dotenv['EMBEDDING_MODEL'],
    ) -> None:
        self.path = path
        self.mode = mode
        self.collection_name = collection_name
        self.EMBEDDING_MODEL = embedding_model

        self.openai = openai
        self.qdrant_client = None
        self.points = None

        self.openai.api_key = _dotenv['OPENAI_APIKEY']

    def instantiate_vector_db(self, autocreate: bool=False) -> None:
        if    self.mode == "localhost":
            qdrant_client = QdrantClient("localhost", port=6333)
        elif  self.mode == "memory":
            qdrant_client = QdrantClient(":memory:")
        elif  self.mode == "hdd":
            qdrant_client = QdrantClient(path=self.path) #"./qdrant"

        try:
            collection_info = qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception as e:
            if not autocreate:
                raise e
            print(f'{e}\n Creating {self.collection_name} collection...')
            self.__create_collection(qdrant_client)
        print(collection_info)

        self.qdrant_client = qdrant_client
        return None
        

    def __create_collection(self, client: QdrantClient) -> None:
        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        return None

    def generate_embeddings(self, chunks: list[str], retry: int=20) -> list:
        #model RPM	TPM
        #text-embedding-ada-002	3000  1 000 000
        #openai.api_key = _dotenv['OPENAI_APIKEY']
        #self.openai.api_key = _dotenv['OPENAI_APIKEY']
        points = []
        for i, chunk in enumerate(chunks, 1):
            #print(f"Embeddings {i} chunk: {chunk}")
            print(f"Embeddings {i}")
            #response = self.openai.Embedding.create(
            #    input=chunk,
            #    model=self.EMBEDDING_MODEL
            #)
            #embeddings = response['data'][0]['embedding']

            for i in range(retry):
                retry_formula = (0.5*(2**i * 100))*0.001
                print(f"Retry {i} {retry_formula} milliseconds")
                time.sleep(retry_formula) if i > 0 else None # backoff strategy in milliseconds
                try:
                    response = self.openai.Embedding.create(
                        input=chunk,
                        model=self.EMBEDDING_MODEL
                    )
                    embeddings = response['data'][0]['embedding']
                    break
                except Exception as e:
                    if i >= retry:
                        raise e

            # new class for "dataframe" based on pointsctruct?
            points.append(PointStruct(id=i, vector=embeddings, payload={"text": chunk}))
            #print("---")
            self.points = points
        return points


    def vdb_insert(self) -> None:
        operation_info = self.qdrant_client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=self.points
        )

        print("Operation info:", operation_info)
        return None
        

    def create_answer_with_context(self, user_input: str, with_context: bool=True) -> str:
        prompt = user_input
        context_json = {}
        if with_context:
            response = self.openai.Embedding.create(
                input=user_input,
                model=self.EMBEDDING_MODEL
            )
            embeddings = response['data'][0]['embedding']

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embeddings, 
                limit=5
            )

            prompt = "Contexto:\n"
            for result in search_results:
                prompt += result.payload['text'] + "\n---\n"
            prompt += f"Pergunta: {user_input}\n---\nResposta:"

            #print("----PROMPT START----")
            #print(prompt)
            #print("----PROMPT END----")

            context_json = {
                "context": search_results,
                "prompt_embedding": embeddings
            }

        completion = openai.ChatCompletion.create(
            model=_dotenv['GENERATOR_MODEL'],
            messages=[
                {"role": "user", "content": prompt}
            ]
            )


        #return completion.choices[0].message.content
        return {
            "user_input": user_input,
            "prompt": prompt,
            "generated": completion.choices[0].message.content,
            **context_json
            #"context": search_results,
            #"prompt_embedding": embeddings
        }
