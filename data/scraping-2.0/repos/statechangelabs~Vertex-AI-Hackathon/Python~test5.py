import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from google.auth import credentials
from google.oauth2 import service_account
from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import ChatModel, TextGenerationModel, InputOutputTextPair
import json  # add this line

from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate, LLMChain

from langchain.vectorstores import MatchingEngine
from langchain.embeddings import VertexAIEmbeddings


with open(
    "vertex-hackathon-creds.json", encoding="utf-8"
) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
    service_account_info = json.load(f)
    project_id = service_account_info["project_id"]


our_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)
embeddings = VertexAIEmbeddings(credentials=our_credentials, project=project_id)

vector_store = MatchingEngine.from_components(
    project_id="<my_project_id>",
    region="<my_region>",
    gcs_bucket_uri="<my_gcs_bucket>",
    index_id="<my_matching_engine_index_id>",
    endpoint_id="<my_matching_engine_endpoint_id>",
    embedding=embeddings,
)





def create_deploy_endpoint():
    # print("Creating index ...")
    # my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    #     credentials = our_credentials,
    #     display_name="vertex-hackathon-index",
    #     contents_delta_uri="gs://vertexai-hackathon-embeddings/batch_root",
    #     dimensions=768,
    #     approximate_neighbors_count=150,
    #     distance_measure_type="COSINE_DISTANCE",
    #     leaf_node_embedding_count = 5000,
    #     leaf_nodes_to_search_percent = 3,
    #     location = "us-central1",
    # )
    
    # print("Index created:", my_index)
    # print("Creating index endpoint ...")
    
    # my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    #     credentials=our_credentials,
    #     display_name=f"public-test-endpoint",
    #     public_endpoint_enabled = True,
    # )
    # print("Index endpoint created:", my_index_endpoint)
    # print("Deploying index to endpoint ...")
    
    # my_index_endpoint = my_index_endpoint.deploy_index(
    # index=my_index,
    # deployed_index_id='4847060671608651776'
    # )
    # print("Deployed indices are:", my_index_endpoint.deployed_indexes)
    # print("Deployed index endpoint is:", my_index_endpoint)
    print("Project ID is->>>>>>>>>>>> ",project_id)
    my_index = aiplatform.MatchingEngineIndex(index_name='projects/287955880258/locations/us-central1/indexes/279847699501547520', credentials=our_credentials, location='us-central1')
    # print(my_index.deployed_indexes) 
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name='6715491567013986304', credentials=our_credentials, project=project_id, location='us-central1')
    index_endpoint.deploy_index(index=my_index, deployed_index_id='test_deployed_index')
    
    return #my_index_endpoint

create_deploy_endpoint()