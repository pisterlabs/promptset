import weaviate
from openai import OpenAI

from democratic_agent.config.config import Config


# TODO: Define more schemas -> User info, tool, episode (previous tasks)...
DEF_SCHEMA = {
    "classes": [
        {
            "class": "UserInfo",
            "description": "User information",
            "properties": [
                {
                    "name": "user_name",
                    "dataType": ["text"],
                    "description": "The name of the user",
                },
                {
                    "name": "info",
                    "dataType": ["text"],
                    "description": "The information of the user",
                },
            ],
        },
        {
            "class": "Tool",
            "description": "Tool information",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "The name of the tool",
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "The description of the tool",
                },
            ],
        },
    ]
}


class WeaviateDB(object):
    def __init__(self):
        weaviate_key = Config().weaviate_key
        self.openai_client = OpenAI()

        if weaviate_key:
            # Run on weaviate cloud service
            auth = weaviate.auth.AuthApiKey(api_key=weaviate_key)
            self.client = weaviate.Client(
                url=Config().weaviate_url,
                auth_client_secret=auth,
                additional_headers={
                    "X-OpenAI-Api-Key": Config().openai_api_key,
                },
            )
        else:
            # Run locally"
            self.client = weaviate.Client(
                url=f"{Config().local_weaviate_url}:{Config().weaviate_port}"
            )
        # self.client.schema.delete_all()
        self._create_schema()

    def _create_schema(self):
        # Check if classes in the schema already exist in Weaviate
        for class_definition in DEF_SCHEMA["classes"]:
            class_name = class_definition["class"]
            print("Creating class: ", class_name)
            try:
                if not self.client.schema.contains(class_definition):
                    # Class doesn't exist, so we attempt to create it
                    self.client.schema.create_class(class_definition)
            except Exception as err:
                print(f"Unexpected error {err=}, {type(err)=}")

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.openai_client.embeddings.create(
                input=[text], model="text-embedding-ada-002"
            )
            .data[0]
            .embedding
        )

    def _get_relevant(
        self,
        vector,
        class_name,
        fields,
        where_filter=None,
        num_relevant=2,
    ):
        try:
            query = (
                self.client.query.get(class_name, fields)
                .with_near_vector(vector)
                .with_limit(num_relevant)
                .with_additional(["certainty", "id"])
            )
            if where_filter:
                query.with_where(where_filter)
            results = query.do()

            if len(results["data"]["Get"][class_name]) > 0:
                return results["data"]["Get"][class_name]
            else:
                return None

        except Exception as err:
            print(f"Unexpected error {err=}, {type(err)=}")
            return None

    def search(self, user_name: str, query: str, num_relevant=2, certainty=0.7):
        query_vector = self.get_ada_embedding(query)
        # Get the most similar content
        user_filter = {
            "path": ["user_name"],
            "operator": "Equal",
            "valueText": user_name,
        }
        most_similar_contents = self._get_relevant(
            vector=({"vector": query_vector}),  # TODO: add "certainty": certainty
            class_name="UserInfo",
            fields=["user_name", "info"],
            where_filter=user_filter,
            num_relevant=num_relevant,
        )
        return most_similar_contents

    def search_tool(self, query: str, num_relevant=2, certainty=0.7):
        query_vector = self.get_ada_embedding(query)
        # Get the most similar content
        most_similar_contents = self._get_relevant(
            vector=({"vector": query_vector}),  # TODO: add "certainty": certainty
            class_name="Tool",
            fields=["name", "description"],
            num_relevant=num_relevant,
        )
        return most_similar_contents

    def store_tool(self, name: str, description: str):
        """Store a tool in the database in case it doesn't exist yet"""

        try:
            query = (
                self.client.query.get("Tool", ["name", "description"])
                .with_limit(1)
                .with_additional(["certainty", "id"])
            )
            tool_name_filter = {
                "path": ["name"],
                "operator": "Equal",
                "valueText": name,
            }
            query.with_where(tool_name_filter)
            results = query.do()

            if len(results["data"]["Get"]["Tool"]) > 0:
                return results["data"]["Get"]["Tool"][0]["_additional"]["id"]
            else:
                info_vector = self.get_ada_embedding(description)
                tool_uuid = self.client.data_object.create(
                    data_object={
                        "name": name,
                        "description": description,
                    },
                    class_name="Tool",
                    vector=info_vector,
                )
                return tool_uuid
        except Exception as err:
            print(f"Unexpected error {err=}, {type(err)=}")
            return None

    def store(
        self,
        user_name: str,
        info: str,
    ):
        # TODO: If the object doesn't exist, proceed with creating a new one
        info_vector = self.get_ada_embedding(info)
        user_info_uuid = self.client.data_object.create(
            data_object={
                "user_name": user_name,
                "info": info,
            },
            class_name="UserInfo",
            vector=info_vector,
        )
        return user_info_uuid
