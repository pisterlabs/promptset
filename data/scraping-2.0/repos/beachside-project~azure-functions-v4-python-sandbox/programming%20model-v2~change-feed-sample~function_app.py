import azure.functions as func
import logging
import openai
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
DEPLOYMENT_ID = "text-embedding-ada-002"

search_client = SearchClient(os.getenv("COGNITIVE_SEARCH_ENDPOINT"),
                             os.getenv("COGNITIVE_SEARCH_INDEX_NAME"), 
                             AzureKeyCredential(os.getenv("COGNITIVE_SEARCH_ADMIN_KEY")))

def get_embedding(text: str):
    embedding = openai.Embedding.create(input=text, deployment_id=DEPLOYMENT_ID)["data"][0]["embedding"]
    return embedding

app = func.FunctionApp()

@app.cosmos_db_trigger(arg_name="items", container_name="azure",
                        database_name="handson", connection="COSMOS_CONNECTION")  
def indexer(items: func.DocumentList):
    documents_to_upsert =[]
    for item in items:
        documents_to_upsert.append({
            "id": item["id"],
            "title": item["title"],
            "category": item["category"],
            "content": item["content"],
            "contentVector": get_embedding(item["content"])
        })
    
    search_client.merge_or_upload_documents(documents_to_upsert)
    logging.info(f"{documents_to_upsert.count} document(s) uploaded")