# using os for env variables
import os

# openai for getting search query embeddings
import openai

# rockset's python module
from rockset import *
from rockset.models import *

# i'm so pretty, oh so pretty
from pprint import pprint

# doing time capture the easy but wrong way
from datetime import datetime

# handy openai globals
openai_org = os.getenv("OPENAI_ORG")
openai_api_key = os.getenv("OPENAI_API_KEY")

# and some Rockset globals
ROCKSET_API_KEY = os.getenv('ROCKSET_API_KEY')
region = Regions.usw2a1

# function for getting user input, returns a dict of inputs
def get_inputs():
    print("Enter your search text:", end=' ')
    search_query = input()

    print("Enter your min price:", end=' ')
    min_price = input()

    print("Enter your max price:", end=' ')
    max_price = input()

    print("Enter your brand:", end=' ')
    brand = input()

    print("Enter your result limit:", end=' ')
    limit = input()

    return {
        "search_query": search_query, 
        "min_price": min_price, 
        "max_price": max_price, 
        "brand": brand, 
        "limit": limit
    }

# function for getting openai embedding, returns embedding (array of floats)
def get_openai_embedding(inputs, org, api_key):
    openai.organization = org
    openai.api_key = api_key

    openai_start = (datetime.now())
    response = openai.Embedding.create(
        input=inputs["search_query"], 
        model="text-embedding-ada-002"
        )
    search_query_embedding = response["data"][0]["embedding"]  
    openai_end = (datetime.now())
    elapsed_time = openai_end - openai_start

    print("\nOpenAI Elapsed time: " + str(elapsed_time.total_seconds()))
    print("Embedding for \"space wars\" looks like " + str(search_query_embedding)[0:100] + "...")

    return search_query_embedding

# function for running Rockset search queries, prints a bunch of stuff
def get_rs_results(inputs, region, api_key, search_query_embedding):
    print("\nRunning Rockset Queries...")
    
    # Create an instance of the Rockset client
    rs = RocksetClient(api_key=api_key, host=region)

    # Execute Query Lambda By Version
    rockset_start = (datetime.now())
    api_response = rs.QueryLambdas.execute_query_lambda_by_tag(
        workspace="confluent_webinar",
        query_lambda="find_related_games_vs",
        tag="latest",
        parameters=[
            {
                "name": "embedding",
                "type": "array",
                "value": str(search_query_embedding)
            },
            {
                "name": "min_price",
                "type": "float",
                "value": inputs["min_price"]
            },
            {
                "name": "max_price",
                "type": "float",
                "value": inputs["max_price"]
            },
            {
                "name": "brand",
                "type": "string",
                "value": inputs["brand"]
            },
            {
                "name": "limit",
                "type": "int",
                "value": inputs["limit"]
            }
        ]
    )
    rockset_end = (datetime.now())
    elapsed_time = rockset_end - rockset_start
    print("\nVector Search Elapsed time: " + str(elapsed_time.total_seconds()))

    print("\nVector Search result:")
    for record in api_response["results"]:
        pprint(record["title"])

    # now let's do the full text search approach
    # we'll split the search query text into the first two terms
    (term1, term2) = inputs["search_query"].split()

    rockset_start = (datetime.now())
    api_response = rs.QueryLambdas.execute_query_lambda_by_tag(
        workspace="confluent_webinar",
        query_lambda="find_related_games_fts",
        tag="latest",
        parameters=[
            {
                "name": "term1",
                "type": "string",
                "value": str(term1)
            },
            {
                "name": "term2",
                "type": "string",
                "value": str(term2)
            },
            {
                "name": "min_price",
                "type": "float",
                "value": inputs["min_price"]
            },
            {
                "name": "max_price",
                "type": "float",
                "value": inputs["max_price"]
            },
            {
                "name": "brand",
                "type": "string",
                "value": inputs["brand"]
            },
            {
                "name": "limit",
                "type": "int",
                "value": inputs["limit"]
            }
        ]
    )
    rockset_end = (datetime.now())
    elapsed_time = rockset_end - rockset_start
    print("\nFTS Elapsed time: " + str(elapsed_time.total_seconds()))

    print("\nFTS result:")
    for record in api_response["results"]:
        pprint(record["title"])

def main():
    print("What kind of video game are you looking for?")
    inputs = get_inputs()
    search_query_embedding = get_openai_embedding(inputs, openai_org, openai_api_key)
    get_rs_results(inputs, region, ROCKSET_API_KEY, search_query_embedding)

if __name__ == "__main__":
    main()