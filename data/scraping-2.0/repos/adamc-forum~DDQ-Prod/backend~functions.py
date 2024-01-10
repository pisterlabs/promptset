from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from openai import AzureOpenAI

from constants import (
    SUBSCRIPTION_ID,
    OPENAI_API_VERSION,
    OPENAI_API_KEY,
    OPENAI_API_ENDPOINT, 
    RG_NAME,
    ACCOUNT_NAME,
    CONNECTION_STRING,
    DATABASE_NAME, 
    COLLECTION_NAME
)

from database import (
    DatabaseClient
)

def get_service_management_client():
    return CognitiveServicesManagementClient (
        credential=DefaultAzureCredential(), 
        subscription_id=SUBSCRIPTION_ID
    )

def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_API_ENDPOINT 
    )

def get_db_client() -> DatabaseClient:
    return DatabaseClient(
        connection_string=CONNECTION_STRING, 
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME
    )

def get_models() -> tuple[str, str]:
    service_management_client = get_service_management_client()
    deployments = service_management_client.deployments.list(RG_NAME, ACCOUNT_NAME)

    deployment_models = [deployment.name for deployment in deployments]

    embedding_model = "text-embedding-ada-002"
    completion_model = "gpt-35-turbo-16k"

    for deployment_model in deployment_models:
        embedding_model = deployment_model if "embedding" in deployment_model.lower() else embedding_model
        completion_model = deployment_model if "completion" in deployment_model.lower() else completion_model
    
    return (embedding_model, completion_model)