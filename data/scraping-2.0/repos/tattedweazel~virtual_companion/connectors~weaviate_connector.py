from tools.secret_squirrel import SecretSquirrel
from langchain.vectorstores import Weaviate
import weaviate


class WeaviateConnector():

    def __init__(self):
        self.creds = SecretSquirrel().stash
        self.auth_config = weaviate.AuthApiKey(api_key=self.creds['weaviate_api_key'])

    
    def get_client(self):
        return weaviate.Client(
            url=self.creds["weaviate_url"],
            auth_client_secret=self.auth_config,
            additional_headers={
                'X-OpenAI-Api-Key': self.creds["open_ai_api_key"]
            }
        )