import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch

def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_llm = Bedrock(
        model_id=model_version_id, 
        client=bedrock_client,
        model_kwargs={'temperature': 0}
        )
    return bedrock_llm

def get_opensearch_endpoint(domain_name='rag'):
    client = boto3.client('es')
    response = client.describe_elasticsearch_domain(
        DomainName=domain_name
    )
    return response['DomainStatus']['Endpoint']

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client


def get_secret(secret_prefix):
    client = boto3.client('secretsmanager')
    secret_arn = locate_secret_arn(secret_prefix, client)
    secret_value = client.get_secret_value(SecretId=secret_arn)
    return secret_value['SecretString']
    
    
def locate_secret_arn(secret_tag_value, client):
    response = client.list_secrets(
        Filters=[
            {
                'Key': 'tag-key',
                'Values': ['Name']
            },
            {
                'Key': 'tag-value',
                'Values': [secret_tag_value]
            }
        ]
    )
    return response['SecretList'][0]['ARN']


def create_opensearch_vector_search_client(index_name, opensearch_password, bedrock_embeddings_client, opensearch_endpoint, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(index_name, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch