
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import openai


class CosmosDbService:
    def __init__(self, endpoint, key, database_name, container_name, partition_key="/conversation-id"):
        self.client = CosmosClient(endpoint, key)
        self.database_client = self.client.get_database_client(database_name)
        try:
            self.container = self.database_client.get_container_client(container_name)
            # Try to read the container to see if it exists
            self.container.read()
        except exceptions.CosmosResourceNotFoundError:
            # If the container does not exist, create it
            self.container = self.database_client.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path=partition_key)
            )

    def add_item(self, item):
        self.container.upsert_item(item)

class OpenAIService:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_message(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].text.strip()
