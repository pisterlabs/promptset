import pprint as pp
import numpy as np
import chromadb
import uuid
from enum import Enum

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# client = chromadb.Client()
client = chromadb.PersistentClient(path="./")
client.delete_collection("mvp")
# ChromaDB how to delete a collection
collection = client.get_or_create_collection("mvp")

class ModelType(Enum):
    GPT4 = "gpt-4"

openAIClient = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openAIClient.embeddings.create(input=[text], model=model).data[0].embedding


class EvalDB:
    def __init__(self, path, collection_name="mvp"):
        self.client = chromadb.PersistentClient(path=path)
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embed = get_embedding

    def add(self, model, datapoint):
        self.collection.add(
            embeddings=self.embed(model.get_query(datapoint)),
            metadatas={**model.evaluate(datapoint), "model_type": model.model_type},
            ids=str(uuid.uuid4()),
        )

    def query(self, model, datapoint, n_results=3):
        return self.collection.query(
            query_embeddings=[self.embed(model.get_query(datapoint))],
            n_results=n_results,
            where={"model_type": {"$eq": model.model_type}},
        )

def test():
    class Model:
        def __init__(self, increment):
            self.model_type = ModelType.GPT4.value + "-" + str(increment)
            self.increment = increment
            self.i = 0

        def get_query(self, datapoint=""):
            return datapoint

        def evaluate(self, datapoint=""):
            self.i += self.increment
            return {
                "performance": self.i,
                "metadata": datapoint,
            }

    model1 = Model(0.005)
    model2 = Model(0.01)


    db = EvalDB("./")

    dataset = ["How many mangos are there?", "How many apples are there?", "What is the best bike to buy?", "Who is the President of the US?", "Summarize the number of apples in the world in two sentences or fewer", "How many oranges are there in the whole wide world?", "Apple is going up in the world."]
    for datapoint in dataset:
        print(f"Adding {datapoint} to DB")
        db.add(model1, datapoint)
        db.add(model2, datapoint)

    query = "What is the total number of apples in the world?"

    nns1 = db.query(model1, query)
    nns2 = db.query(model2, query)

    pp.pprint(nns1)
    print("\n\n")
    pp.pprint(nns2)

    performance1 = np.mean([x[0]["performance"]
                            for x in nns1['metadatas']])

    performance2 = np.mean([x[0]["performance"]
                            for x in nns2['metadatas']])

    print(f"Model 1 performance for query {query} is: {performance1}")
    print(f"Model 2 performance for query {query} is: {performance2}")

if __name__ == "__main__":
    test()
