import yaml
import requests
import json
import requests
import logging 
import boto3
import botocore
import os
from requests.auth import HTTPBasicAuth
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from langchain.document_loaders import UnstructuredPDFLoader 
from langchain.text_splitter import CharacterTextSplitter


logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

TEXT_EMBEDDING_MODEL_ENDPOINT_NAME = os.environ.get('TEXT_EMBEDDING_MODEL_ENDPOINT_NAME')
sagemaker_client = boto3.client('runtime.sagemaker')

es_username = os.environ.get('VECTOR_DB_USERNAME')
es_password = os.environ.get('VECTOR_DB_PASSWORD')

domain_endpoint = os.environ.get('VECTOR_DB_ENDPOINT')
domain_index = os.environ.get('VECTOR_DB_INDEX')

URL = f'{domain_endpoint}/{domain_index}'
logger.info(f'URL for OpenSearch index = {URL}')

mapping = {
    'settings': {
        'index': {
            'knn': True  # Enable k-NN search for this index
        }
    },
    'mappings': {
        'properties': {
            'embedding': {  # k-NN vector field
                'type': 'knn_vector',
                'dimension': 4096  # Dimension of the vector
            },
            'passage_id': {
                'type': 'keyword'
            },
            'passage': {
                'type': 'text'
            }
        }
    }
}

# Check if the index exists using an HTTP HEAD request
response = requests.head(URL, auth=HTTPBasicAuth(es_username, es_password))

# If the index does not exist (status code 404), create the index
if response.status_code == 404:
    response = requests.put(URL, auth=HTTPBasicAuth(es_username, es_password), json=mapping)
    logger.info(f'Index created: {response.text}')
else:
    logger.error('Index already exists!')
    
    
    
# loader = UnstructuredPDFLoader("./sop_cd.pdf")
# Initialize AWS S3 client
s3_client = boto3.client('s3')

# Specify the S3 bucket and object key
s3_bucket = os.environ.get('S3_BUCKET')
s3_key = os.environ.get('S3_KEY')

# Specify the local path to save the downloaded PDF
local_pdf_path = f'/tmp/{s3_key}'

# Download the PDF from S3
try:
    s3_client.download_file(s3_bucket, s3_key, local_pdf_path)
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == 'NoSuchKey':
        print(f"The object '{s3_key}' does not exist in the bucket '{s3_bucket}'.")
    elif e.response['Error']['Code'] == 'AccessDenied':
        print(f"Access to the object '{s3_key}' in the bucket '{s3_bucket}' is denied.")
    else:
        print(f"An error occurred: {e}")

# Initialize the PDF loader
loader = UnstructuredPDFLoader(local_pdf_path)

loaded_docs = loader.load()

# Splitting documents into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(loaded_docs)

# Find the length (number of chunks) in chunked_docs
num_chunks = len(chunked_docs)
print(f"Number of chunks: {num_chunks}")

# Print the content of each chunk
for chunk in chunked_docs:
    chunk_content = chunk
    print("*************************")
    print(chunk_content)
    

def process_chunks(chunked_docs):
    # Replace `chunk_iterator(CHUNKS_DIR_PATH)` with `chunked_docs` and remove tqdm if not needed.
    for i, chunk in tqdm(enumerate(chunked_docs)):
        
   
        chunk_text = chunk.page_content
        
        payload = {'text_inputs': [chunk_text]}
        payload = json.dumps(payload).encode('utf-8')

        response = sagemaker_client.invoke_endpoint(
            EndpointName=TEXT_EMBEDDING_MODEL_ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )

        model_predictions = json.loads(response['Body'].read())
        embedding = model_predictions['embedding'][0]

        document = {
            'passage_id': i,  # Using the loop index as the chunk_id
            'passage': chunk_text,
            'embedding': embedding
        }

        response = requests.post(
            f'{URL}/_doc/{i}',
            auth=HTTPBasicAuth(es_username, es_password),
            json=document
        )

        if response.status_code not in [200, 201]:
            print(document)
            logger.error(response.status_code)
            logger.error(response.text)
            break
            
process_chunks(chunked_docs)