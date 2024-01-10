# import modules
from getpass import getpass
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from urllib.request import urlopen
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


####

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id
ELASTIC_CLOUD_ID = "748dd50a1c52448984c85adbe58a891d:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ5NTI4NDdlNmY5MWM0OWM1OWFjNzhiZmQzOTY2YjY0YiRiYzY4ZjMwZmFkNWU0MzA1OWNmOTVmMDg0MzY1MzI4MQ=="
print("Got Elastic Cloud ID vector store")

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key
ELASTIC_API_KEY = "essu_U1ZCeFRIZEpkMEoxUzI1SFJESlRjM0JqVUY4NlVuVnZSMlJ2WTJWU09IVlRUSFpRZDJGRFFYUXRadz09AAAAAJCJYMU="

print("Got Elastic Key vector store")
# https://platform.openai.com/api-keys
OPENAI_API_KEY = "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"

print("Creating vector store")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

print("Creating vector store")
vector_store = ElasticsearchStore(
    es_cloud_id=ELASTIC_CLOUD_ID,
    es_api_key=ELASTIC_API_KEY,
    index_name= "workplace_index",
    embedding=embeddings
)

print(vector_store)