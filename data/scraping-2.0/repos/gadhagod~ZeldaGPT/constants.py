from os import getenv
from datetime import datetime
from rockset import RocksetClient, exceptions
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Rockset as RocksetStore
from queries import ingest_tranformation

is_production = getenv("ENVIRONMENT") == "prod"
rockset_api_server = getenv("ROCKSET_API_SERVER")
rockset_api_key = getenv("ROCKSET_API_KEY")
openai_api_key = getenv("OPENAI_API_KEY")

rockset = RocksetClient(rockset_api_server, rockset_api_key)

class Collection:
    def __init__(self, workspace, name):
        self.workspace = workspace
        self.name = name
    
    def exists(self):
        try:
            rockset.Collections.get(collection=self.name)
        except exceptions.NotFoundException:
            return False
        return True

    def is_ready(self):
        return rockset.Collections.get(collection=self.name).data.status == "READY"

    def delete(self):
        print(f"Deleting collection \"{self.workspace}.{self.name}\"")
        rockset.Collections.delete(collection=self.name)
        
    def create(self):
        print(f"Creating collection \"{self.workspace}.{self.name}\"")
        rockset.Collections.create_s3_collection(name=self.name, field_mapping_query=ingest_tranformation())
        
    def add_doc(self, doc):
        rockset.Documents.add_documents(
            collection=self.name,
            data=[doc]
        )
    
    @property
    def created_at(self):
        return datetime.strptime(rockset.Collections.get(collection=self.name).data.created_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%x")
    

embeddingCollection = Collection("commons", "zeldagpt")
questionCollection = Collection("commons", "zeldagpt-questions") if is_production else None

openai = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-ada-002"
)
store = RocksetStore(
    rockset,
    openai,
    embeddingCollection.name,
    "text",
    "embedding"
)