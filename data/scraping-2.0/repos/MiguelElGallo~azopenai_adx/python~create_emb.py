# https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-python?tabs=azure-portal%2Cpasswordless%2Clinux%2Csign-in-azure-cli%2Csync#authenticate-the-client

import openai
import argparse
import pandas as pd
import tiktoken
import re
from azure.identity import AzureDeveloperCliCredential
import os
import json
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import time
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey

args = argparse.Namespace(verbose=False)

df = pd.DataFrame()
open_ai_token_cache = {}
CACHE_KEY_TOKEN_CRED = 'openai_token_cred'
CACHE_KEY_CREATED_TIME = 'created_time'
CACHE_KEY_TOKEN_TYPE = 'token_type'

#Embedding batch support section
SUPPORTED_BATCH_AOAI_MODEL = {
    'text-embedding-ada-002': {
        'token_limit' : 8100,
        'max_batch_size' : 16
    }
}

def refresh_openai_token():
    """
    Refresh OpenAI token every 5 minutes
    """
    if CACHE_KEY_TOKEN_TYPE in open_ai_token_cache and open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] == 'azure_ad' and open_ai_token_cache[CACHE_KEY_CREATED_TIME] + 300 < time.time():
        token_cred = open_ai_token_cache[CACHE_KEY_TOKEN_CRED]
        openai.api_key = token_cred.get_token("https://cognitiveservices.azure.com/.default").token
        open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()

def load_csv(path):
    global df
    # Load data into pandas DataFrame from "/lakehouse/default/" + "Files/masterdata/product_hier/dairy_products.csv"
    df = pd.read_csv(path + "dairy_products.csv")
    print(df.head())

    # Create new column text_to_embedd
    df['text_to_embedd'] = "Category: " + df['product_hier'].map(str) + ". " + "Description:" + df['description'].map(str)
    print(df.head())


    # Now let's prepare to call the OpenAI model to generate and embedding
    # s is input text , and all special characters are removed. Including double spaces, double dots, etc.
    def normalize_text(s, sep_token = " \n "):
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        return s

    # We need to use tokenizer to get the number of tokens before calling the embedding
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # remove double spaces, dots, etc.
    df['text_to_embedd'] = df['text_to_embedd'].apply(lambda x : normalize_text(x))
    # add new column with number of tokens
    df['n_tokens'] = df["text_to_embedd"].apply(lambda x: len(tokenizer.encode(x)))


    #Now we call the OpenAI  model for getting the embeddings
    azd_credential = AzureDeveloperCliCredential() 

    openai.api_base = f"https://{args.openaiservice}.openai.azure.com"    
    openai.api_type = "azure_ad"
    openai.api_key = azd_credential.get_token("https://cognitiveservices.azure.com/.default").token
    openai.api_version = "2022-12-01"
    open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
    open_ai_token_cache[CACHE_KEY_TOKEN_CRED] = azd_credential
    open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] = "azure_ad"
    

    refresh_openai_token()
    print("OpenAI token refreshed")
    print("Using OpenAI service: " + openai.api_base)
    print("Using OpenAI deployment: " + args.openaideployment)

    # df['vector'] = df["text_to_embedd"].apply(lambda x : get_embedding(x, engine = args.openaideployment)) # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    df['vector'] = df["text_to_embedd"].apply(lambda x : openai.Embedding.create(engine=args.openaideployment, input=x)) 

    print(df.head())



    return 0

def create_record(df):
    hier_record = { 'id':df['ID'],
                    'partitionKey':df['ID'],
                    'product_hier':df['product_hier'],
                    'description':df['description'],
                    'vector':df['vector']["data"][0]["embedding"] #OpenAI returns a JSON object with a "data" array. We need the first element of the array and the "embedding" element of the object
                    }


    return hier_record

def write_to_cosmosdb(container,row):
    container.upsert_item(body=row)
    return 0 

def delete_from_cosmosdb(container):
    # Delete all records from Cosmos DB using delete_all_items_by_partition_key
    # container.delete_all_items_by_partition_key(partition_key = 'ID') In preview 
    for item in container.query_items(query='SELECT * FROM container1',
                                  enable_cross_partition_query=True):
        print("Deleting from CosmosDB item: " + item['id'])
        container.delete_item(item, partition_key=item['id'])

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads a CSV and creates embeddings for each row.",
        epilog="Example: create_emb.py'..\data\*'"
        )
    parser.add_argument("files", help="Files to be processed")
    parser.add_argument("--openaiservice", help="Name of the Azure OpenAI service used to compute embeddings")
    parser.add_argument("--openaideployment", help="Name of the Azure OpenAI model deployment for an embedding model ('text-embedding-ada-002' recommended)")
    parser.add_argument("--openaimodelname", help="Name of the Azure OpenAI embedding model ('text-embedding-ada-002' recommended)")
    args = parser.parse_args()
    # Cosmos DB connection
    ## Let's read the environment variables set by AZD
    endpoint = os.environ["COSMOS_ENDPOINT"]

    # Load data into pandas DataFrame and create embeddings for each row using OpenAI
    load_csv(args.files)

    # In infra/core/storage/cosmosdb.bicep we have defined the Cosmos DB account and database and container
    # Database is called "database1"
    # Container is called "container1"
    # The partition key for the container is "/partitionKey"
    # Connect to Cosmos DB
    ENDPOINT = os.environ["COSMOS_ENDPOINT"]
    print("Cosmos Endpoint: " + ENDPOINT)
    
    credential = AzureDeveloperCliCredential()
    client = CosmosClient(ENDPOINT, credential)
    database_name = 'database1'
    database = client.get_database_client(database_name)
    container_name = 'container1'
    container = database.create_container_if_not_exists(id=container_name,
                                                      partition_key=PartitionKey(path='/partitionKey', kind='Hash'))
    

    # We delete all records from Cosmos DB, to make it idempotent
    delete_from_cosmosdb(container)
    try:
        for i in range(len(df)):
            record = create_record(df.iloc[i])
            write_to_cosmosdb(container,record)
            print("Record " + str(i) + " written to Cosmos DB")


    except exceptions.CosmosHttpResponseError as e:
        print('\nError while writing to cosmos {0}'.format(e.message))
    finally:
        print("\nHierarchy written to Cosmos DB")        

    
    # Now we need an index for the embeddings
