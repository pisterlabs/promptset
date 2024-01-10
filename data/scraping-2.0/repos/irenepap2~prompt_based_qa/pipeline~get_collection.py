import argparse
import requests
import json
import os
import csv
import pipeline.utils as utils

from pipeline.constants import API_BASE_URL, COLLECTION_PATH

def generate_queries(query):
    prompt = utils.construct_prompt(
    filename="search_filter_extract_query_prompt.txt",
    prompt_params_dict={
        "current_dt": utils.get_current_date(),
        "input_query": query,
    }
    )
    print("Getting search, filter and extract queries from OpenAI...")
    return utils.get_model_response(prompt, args.model)["choices"][0]["text"]

def get_collection_with_gold_queries(query_types, args):
    extract_q = query_types["extract_query"]
    search_q = query_types["search_query"]
    filter_q_dict = eval(query_types["filter_query"])
    print(f"Search query: {search_q}")
    print(f"Filter query: {filter_q_dict}")
    print(f"Extract query: {extract_q}")

    filters_dict = {
        "query_string": search_q, 
        "retrieval_method": args.retrieval_method,
        "document_types": "document",
        "with_code": "false",
        "page_size": args.page_size,
        "page": 1,
        **filter_q_dict   
    }

    print("Calling API to get collection of documents...")
    hits, url = call_api(filters_dict, args.url)

    #save collection of documents to json file
    return save_collection(url, hits, query, search_q, extract_q, filter_q_dict, args)

def get_collection(query, args):
    '''
    Get collection of documents from API
    Args:
        query (str): query to search for documents
        args (argparse.Namespace): arguments passed to the script
    ''' 

    response = generate_queries(query)
    search_q, filter_q_dict, extract_q = utils.parse_search_filter_extract_response(response)
    print(f"Search query: {search_q}")
    print(f"Filter query: {filter_q_dict}")
    print(f"Extract query: {extract_q}")

    filters_dict = {
        "query_string": search_q, 
        "retrieval_method": args.retrieval_method,
        "document_types": "document",
        "with_code": "false",
        "page_size": args.page_size,
        "page": 1,
        **filter_q_dict   
    }

    print("Calling API to get collection of documents...")
    hits, url = call_api(filters_dict, args.url)

    #save collection of documents to json file
    print("Saving collection of documents to json file under data/collections...")
    
    return save_collection(url, hits, query, search_q, extract_q, filter_q_dict, args)

def get_queries_from_file(query_file):
    '''
    Get queries from file
    Args:
        query_file (file): csv file containing queries
    Returns:
        queries (dict): dictionary of queries
    '''
    queries = {}
    reader = csv.reader(query_file)
    # skip header
    next(reader)
    for row in reader:
        queries[row[0]] = {
            "query": row[1],
            "search_query": row[2],
            "filter_query": row[3],
            "extract_query": row[4],
        }
    return queries

def get_all_results(params, base_url):
    '''
    Get all results from API page by page
    Args:
        base_url (str): API base url to call
        params (dict): parameters to pass to the base_url
    Returns:
        total_hits (list): list of all hits from API
        response_url (str): url of the last response
    '''
    total_hits = []
    hits, response_url = call_api(params, base_url)
    total_hits.extend(hits)
    while len(hits) > 0 and params["page"] < 10:
        params["page"] += 1
        hits, response_url = call_api(params, base_url)
        total_hits.extend(hits)
    return total_hits, response_url

def call_api(params, base_url):
    '''
    Call API to get collection of documents
    Args:
        base_url (str): API base url to call
        params (dict): parameters to pass to the base_url
    Returns:
        hits (list): list of hits from API
        response_url (str): url of the response
    '''
    response = requests.get(url=base_url, params=params)
    response_url = response.url
    print(response_url)
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}")
    
    response = response.json()
    hits = response["hits"]
    print(f"Number of documents retrieved: {len(hits)}")
    return hits, response_url

def save_collection(url, hits, query, search_q, extract_q, filter_q_dict, args):
    # create data folder if it doesn't exist
    if not os.path.exists(os.path.join(COLLECTION_PATH)):
        os.mkdir(os.path.join(COLLECTION_PATH))
        os.mkdir(os.path.join(COLLECTION_PATH, args.retrieval_method))
    
    with open(os.path.join(COLLECTION_PATH, args.retrieval_method, f"{query}.json"), "w") as f:
        collection = {
            "url": url,
            "query": query,
            "search_query": search_q,
            "extract_query": extract_q,
            "filter_query": filter_q_dict,
            "hits": [{
                "guid" : hit["guid"],
                "uri" : hit["uri"],
                "score" : hit["score"],
                "metadata" : hit["metadata"]
                } for hit in hits],
        }
        json.dump(collection, f, indent=4)    
    return collection        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # either user_query or query_file is required
    group = parser.add_mutually_exclusive_group(required=True)
    # read query from command line
    group.add_argument("--query", type=str, help="Query to search for documents")
    # read query from file
    group.add_argument("--query_file", type=argparse.FileType("r"), help="File containing queries to search for documents")
    # define url of API
    parser.add_argument("--url", type=str, default=API_BASE_URL, help="URL of API to call to get collection of documents")
    # option to save collection of documents to json file
    parser.add_argument("--save_collection", type=bool, default=True, help="Save collection of documents to json file")
    # set number of documents to retrieve
    parser.add_argument("--page_size", type=int, default=20, help="Number of documents to retrieve")
    # choose between knn or keyword retrieval
    parser.add_argument("--retrieval_method", type=str, default="knn", help="Retrieval method to use (knn or keyword)")
    # choose which model to use
    parser.add_argument("--model", type=str, choices=["text-davinci-003", "code-davinci-002", "gpt-3.5-turbo"], required=True, help="Choose between the OpenAI large language models")

    args = parser.parse_args()

    if args.query_file:
        all_queries = get_queries_from_file(args.query_file)    
        for query_id, query_types in all_queries.items():
            query = query_types["query"]

            if not os.path.exists(os.path.join(COLLECTION_PATH, args.retrieval_method, "csv_files")):
                os.mkdir(os.path.join(COLLECTION_PATH, args.retrieval_method, "csv_files"))
            if os.path.exists(os.path.join(COLLECTION_PATH, args.retrieval_method, "csv_files", f"{query_id}_{query}.csv")):
                print(f"Collection for {query_id} already exists. Skipping...")
                continue
            
            collection = get_collection_with_gold_queries(query_types, args)
            hits = collection["hits"]

            # save collection to csv file
            with open(os.path.join(COLLECTION_PATH, args.retrieval_method, "csv_files", f"{query_id}_{query}.csv"), "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["guid", "Title", "URL"])
                for hit in hits:
                    uri = hit["uri"].replace("abs", "pdf")
                    writer.writerow([hit["guid"], hit["metadata"]["title"], uri])
    else:
        query = args.query
        get_collection(query, args)
