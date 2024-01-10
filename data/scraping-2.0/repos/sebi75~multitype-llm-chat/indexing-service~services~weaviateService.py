from weaviate import Client
from openai_service import OpenAIService
import pandas as pd


class WeviateService:
    def __init__(self, client: Client, openai_service: OpenAIService):
        self.client = client
        self.openai_service = openai_service

    def getOrCreateClass(self, className: str):
        try:
            schema = self.client.schema.get()
            if self.contains(schema["classes"], lambda x: x["class"] == className):
                print("Class already exists")
                return
            else:
                class_obj = {"class": className}
                self.client.schema.create_class(class_obj)
        except Exception as e:
            print(e)
            print("Error in getOrCreateClass")

    def search(self, query: str, className: str):
        # get embedding for search_query
        search_query_embedding = self.openai_service.get_embedding(query)
        response = (
            self.client.query
            .get(className, ["text"])
            .with_near_vector({
                "vector": search_query_embedding, })
            .with_limit(5)
            .with_additional(["distance"])
            .do()
        )
        id_capitalized = className.capitalize()
        data = response["data"]["Get"][f"{id_capitalized}"]

        return data

    def delete_object(self, className: str):
        self.client.data_object.delete(className)

    def indexing_save(self, result, chat_id: str, object_id: str, client: Client):
        composite_id = f"{chat_id}-{object_id}"
        self.getOrCreateClass(client, composite_id)
        df = pd.DataFrame(result, columns=["chunk"])
        df["embedding"] = df["chunk"].apply(self.openai_service.get_embedding)

        # batch create data objects
        client.batch.configure(batch_size=100)
        with client.batch as batch:
            for _, row in df.iterrows():
                data_object = {
                    "text": row["chunk"],
                }
                batch.add_data_object(data_object=data_object, class_name=composite_id,
                                      vector=row["embedding"])

    def contains(self, list, isInList):
        for x in list:
            if isInList(x):
                return True
        return False
