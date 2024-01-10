import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
import pandas as pd
import json
from helpers import get_aws_auth, get_embeddings_with_retry


def load_file_and_split_text(filename):
    with open(filename) as file:
        sanitized_text = file.read()

        sanitized_text = re.sub(r'\n{2,}', '####', sanitized_text)
        sanitized_text = re.sub(r'\n{1}', ' ', sanitized_text)
        sanitized_text = re.sub(r'####', '\n\n', sanitized_text)
        sanitized_text = re.sub(r' {1,}', ' ', sanitized_text)
        sanitized_text = re.sub(r'“', '"', sanitized_text)
        sanitized_text = re.sub(r'”', '"', sanitized_text)
        sanitized_text = re.sub(r'[^a-zA-Z0-9?.,;\-!\s\n\'"]', '', sanitized_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap  = 500,
            length_function = len,
            add_start_index = True,
        )
        text_segments = text_splitter.create_documents([sanitized_text])
        text_segments = [t.page_content for t in text_segments]
        return text_segments

def get_embeddings_for_segments_and_store(text_segments, index_name, opensearch_endpoint_url, aws_region, start_idx=0):
    """
    Each of the text segments comes from your text splitter. In here, we get the embeddings from OpenAI and save the
    docs to our DB
    """
    for text_segment in text_segments:
        embeddings = get_embeddings_with_retry(text_segment)
        if embeddings:
            doc = {'text': text_segment, 'embeddings': embeddings }
            load_doc_to_open_search(doc, index_name, opensearch_endpoint_url, aws_region)

def get_embeddings_with_retry(text):
    ctr = 0
    while (ctr < 4):
        ctr += 1
        try:
            embedding = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )['data'][0]['embedding']
            return embedding
        except:
            sleep(1)
    print("Unsuccessful at getting embeddings for:", text)

def load_doc_to_open_search(text_to_embeddings_dict, index_name, opensearch_endpoint_url, region):
    auth=get_aws_auth(region)
    endpoint_url = f'{opensearch_endpoint_url}/{index_name}/_doc'
    headers = {'Content-Type': 'application/json'}
    json_doc = json.dumps(text_to_embeddings_dict)
    put_doc_response = requests.post(endpoint_url, headers=headers, data=json_doc, auth=auth)
    print(f"loaded doc to OpenSearch with status: {put_doc_response.status_code}")
    if (put_doc_response.status_code > 299):
        print("Error posting document:", put_doc_response.text)

def create_opensearch_index_if_not_exists(index_name, opensearch_endpoint_url, region):
    """
       Prerequisite: create an opensearch cluster in aws.
       You can just do it through click-ops as detailed here https://docs.aws.amazon.com/opensearch-service/latest/developerguide/gsgcreate-domain.html
       Except I recommend you pick cheaper settings:
       - No standby
       - Single node
       - 10gb volume
       - cheaper instance type (I picked m6.large)
    """
    index_url = f"{opensearch_endpoint_url}/{index_name}"
    auth=get_aws_auth(region)
    #check if index exists
    index_exists_response = requests.head(
        index_url,
        headers={"Content-Type": "application/json"},
        auth=auth
    )

    if index_exists_response.status_code == 200:
        print(f'Index "{index_name}" already exists. Skipping creation.')
        return
    else:
        print(f'Index "{index_name}" not found, creating...')

    mapping_properties = {
        'text': {'type': 'text'},
        'embeddings': {
            'type': 'knn_vector',
            'dimension': 1536,
            'method': {
                'name': 'hnsw',
                'space_type': 'cosinesimil',
                'engine': 'nmslib'
            }
        }
    }
    index_settings = {
        'settings': {
            'index.knn': True
        },
        'mappings': {
            'properties': mapping_properties
        }
    }
    create_result = requests.put(
        index_url,
        headers={"Content-Type": "application/json"},
        auth=auth,
        json=index_settings
    )

    if create_result.status_code == 200:
        print(f'Index "{index_name}" successfully created.')
    else:
        print(f'Error creating index: {create_result.content}')
        exit()


def load_document_main(index_name, opensearch_endpoint_url, aws_region):
    create_opensearch_index_if_not_exists(index_name, opensearch_endpoint_url, aws_region)

    text_segments = load_file_and_split_text("./Trimmed-Seneca-Morals-of-a-Happy-Life-Benefits-Anger-and-Clemency.txt")

    get_embeddings_for_segments_and_store(
        text_segments,
        index_name,
        opensearch_endpoint_url,
        aws_region
    )

def delete_index_if_exists(index_name, opensearch_endpoint_url, aws_region):
    index_url = f"{opensearch_endpoint_url}/{index_name}"
    auth=get_aws_auth(aws_region)
    #check if index exists
    index_exists_response = requests.head(
        index_url,
        headers={"Content-Type": "application/json"},
        auth=auth
    )

    if index_exists_response.status_code != 200:
        return

    delete_result = requests.delete(
        index_url,
        headers={"Content-Type": "application/json"},
        auth=auth
    )
    print("DELETE result: ", delete_result.status_code, delete_result.content)