import os
import sys

#  this is needed and should matach the path used in the processing job dependency image built dockerfile
#  as in the dockerfile we copy the helper code into this folder so here we need to add its path 
#  so the Python import can find the helper code
sys.path.append('/opt/ml/processing/image_code/')
from embedding_helper import create_sagemaker_embeddings_from_js_model

import glob
import time
import json
import logging
import argparse
import numpy as np
import multiprocessing as mp
from itertools import repeat
from functools import partial
import sagemaker, boto3, json
from typing import List, Tuple
from sagemaker.session import Session
# from credentials import get_credentials
from opensearchpy.client import OpenSearch
from langchain.document_loaders import ReadTheDocsLoader
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth.aws4auth import AWS4Auth


# global constants
MAX_OS_DOCS_PER_PUT = 100
TOTAL_INDEX_CREATION_WAIT_TIME = 60
PER_ITER_SLEEP_TIME = 5
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO, stream=sys.stderr)

region = 'ap-southeast-2'
ssm = boto3.client('ssm', region_name=region)
access_key = ssm.get_parameter(Name='ACCESS_KEY', WithDecryption=True)['Parameter']['Value']
secret_key = ssm.get_parameter(Name='SECRET_KEY', WithDecryption=True)['Parameter']['Value']
service = 'es'

aws4auth = AWS4Auth(access_key, secret_key, region, service)
def check_if_index_exists(index_name:str, region: str, host:str, http_auth) -> OpenSearch:
    aos_client = OpenSearch(
        hosts = [{'host': host.replace("https://", ""), 'port': 443}],
        http_auth = http_auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    exists = aos_client.indices.exists(index_name)
    logger.info(f"index_name={index_name}, exists={exists}")
    return exists

def process_shard(shard, embeddings_model_endpoint_name:str, aws_region:str, os_index_name:str, os_domain_ep:str, os_http_auth) -> int: 
    logger.info(f'Starting process_shard of {len(shard)} chunks.')
    st = time.time()
    embeddings = create_sagemaker_embeddings_from_js_model(embeddings_model_endpoint_name, aws_region)
    docsearch = OpenSearchVectorSearch(index_name=os_index_name,
                                       embedding_function=embeddings,
                                       opensearch_url=os_domain_ep,
                                       timeout = 300,
                                       use_ssl = True,
                                       verify_certs = True,
                                       connection_class = RequestsHttpConnection,
                                       http_auth=os_http_auth)    
    docsearch.add_documents(documents=shard)
    et = time.time() - st
    logger.info(f'Shard completed in {et} seconds.')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensearch-cluster-domain", type=str, default=None)
    parser.add_argument("--opensearch-index-name", type=str, default=None)
    parser.add_argument("--region", type=str, default="ap-southeast-2")
    parser.add_argument("--embeddings-model-endpoint-name", type=str, default=None)
    parser.add_argument("--chunk-size-for-doc-split", type=int, default=200)
    parser.add_argument("--chunk-overlap-for-doc-split", type=int, default=30)
    parser.add_argument("--input-data-dir", type=str, default="/opt/ml/processing/input_data")
    parser.add_argument("--process-count", type=int, default=1)
    parser.add_argument("--create-index-hint-file", type=str, default="_create_index_hint")
    args, _ = parser.parse_known_args()

    logger.info("Received arguments {}".format(args))
    # list all the files
    files = glob.glob(os.path.join(args.input_data_dir, "*.*"))
    logger.info(f"there are {len(files)} files to process in the {args.input_data_dir} folder")



    loader = ReadTheDocsLoader(args.input_data_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=args.chunk_size_for_doc_split,
        chunk_overlap=args.chunk_overlap_for_doc_split,
        length_function=len,
    )

    st = time.time()
    logger.info('Loading documents...')
    docs = loader.load()


    # add a custom metadata field, such as timestamp
    for doc in docs:
        doc.metadata['timestamp'] = time.time()
        doc.metadata['embeddings_model'] = args.embeddings_model_endpoint_name
    chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    et = time.time() - st
    logger.info(f'Time taken: {et} seconds. {len(chunks)} chunks generated') 

    db_shards = (len(chunks) // MAX_OS_DOCS_PER_PUT) + 1
    print(f'Loading chunks into vector store ... using {db_shards} shards') 
    st = time.time()
    shards = np.array_split(chunks, db_shards)
    
    t1 = time.time()

    index_exists = check_if_index_exists(index_name = args.opensearch_index_name, region = args.region, host = args.opensearch_cluster_domain, http_auth = aws4auth)

    embeddings = create_sagemaker_embeddings_from_js_model(args.embeddings_model_endpoint_name, args.region)
    

    if index_exists is False:
        path = os.path.join(args.input_data_dir, args.create_index_hint_file)
        logger.info(f"index {args.opensearch_index_name} does not exist but {path} file is present so will create index")
        # by default langchain would create a k-NN index and the embeddings would be ingested as a k-NN vector type
        docsearch = OpenSearchVectorSearch.from_documents(index_name=args.opensearch_index_name,
                                                        documents=shards[0],
                                                        embedding=embeddings,
                                                        opensearch_url=args.opensearch_cluster_domain,
                                                        timeout = 300,
                                                        use_ssl = True,
                                                        verify_certs = True,
                                                        connection_class = RequestsHttpConnection,
                                                        http_auth=aws4auth
                                                        )

        # we now need to start the loop below for the second shard
        shard_start_index = 1 
    else:
        logger.info(f"index={args.opensearch_index_name} does exists, going to call add_documents")
        shard_start_index = 0

    with mp.Pool(processes = args.process_count) as pool:
        results = pool.map(partial(process_shard,
                                   embeddings_model_endpoint_name=args.embeddings_model_endpoint_name,
                                   aws_region=args.region,
                                   os_index_name=args.opensearch_index_name,
                                   os_domain_ep=args.opensearch_cluster_domain,
                                   os_http_auth=aws4auth),
                           shards[shard_start_index:])
    t2 = time.time()
    logger.info(f'run time in seconds: {t2-t1:.2f}')
    logger.info("all done")
    