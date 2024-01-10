import json
import os
import sys
import boto3
import numpy as np
import time
import botocore

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.vectorstores import OpenSearchVectorSearch

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.load.dump import dumps

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww


boto3_bedrock = bedrock.get_bedrock_client(
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)


# Directory name
directory_name = "files"

# File name
file_name = "car_manual.pdf"

# Constructing the full file path using os.path.join
file_path = os.path.join(directory_name, file_name)


loader = PyPDFDirectoryLoader("./files/")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)


avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(
    documents
)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(f"Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.")
print(f"After the split we have {len(docs)} documents more than the original {len(documents)}.")
print(
    f"Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters."
)

try:
    
    sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
    modelId = bedrock_embeddings.model_id
    print("Embedding model Id :", modelId)
    print("Sample embedding of a document chunk: ", sample_embedding)
    print("Size of the embedding: ", sample_embedding.shape)

except ValueError as error:
    if  "AccessDeniedException" in str(error):
        print(f"\x1b[41m{error}\
        \nTo troubeshoot this issue please refer to the following resources.\
         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
        class StopExecution(ValueError):
            def _render_traceback_(self):
                pass
        raise StopExecution        
    else:
        raise error
        

vector_store_name = 'digital-assistant-rag'
index_name = "digital-assistant-rag-index"
encryption_policy_name = "digital-assistant-rag-ep"
network_policy_name = "digital-assistant-rag-np"
access_policy_name = 'digital-assistant-rag-ap'
identity = boto3.client('sts').get_caller_identity()['Arn']

aoss_client = boto3.client('opensearchserverless')

security_policy = aoss_client.create_security_policy(
    name = encryption_policy_name,
    policy = json.dumps(
        {
            'Rules': [{'Resource': ['collection/' + vector_store_name],
            'ResourceType': 'collection'}],
            'AWSOwnedKey': True
        }),
    type = 'encryption'
)

network_policy = aoss_client.create_security_policy(
    name = network_policy_name,
    policy = json.dumps(
        [
            {'Rules': [{'Resource': ['collection/' + vector_store_name],
            'ResourceType': 'collection'}],
            'AllowFromPublic': True}
        ]),
    type = 'network'
)

collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')

while True:
    status = aoss_client.list_collections(collectionFilters={'name':vector_store_name})['collectionSummaries'][0]['status']
    if status in ('ACTIVE', 'FAILED'): break
    time.sleep(10)

access_policy = aoss_client.create_access_policy(
    name = access_policy_name,
    policy = json.dumps(
        [
            {
                'Rules': [
                    {
                        'Resource': ['collection/' + vector_store_name],
                        'Permission': [
                            'aoss:CreateCollectionItems',
                            'aoss:DeleteCollectionItems',
                            'aoss:UpdateCollectionItems',
                            'aoss:DescribeCollectionItems'],
                        'ResourceType': 'collection'
                    },
                    {
                        'Resource': ['index/' + vector_store_name + '/*'],
                        'Permission': [
                            'aoss:CreateIndex',
                            'aoss:DeleteIndex',
                            'aoss:UpdateIndex',
                            'aoss:DescribeIndex',
                            'aoss:ReadDocument',
                            'aoss:WriteDocument'],
                        'ResourceType': 'index'
                    }],
                'Principal': [identity],
                'Description': 'Easy data policy'}
        ]),
    type = 'data'
)

host = collection['createCollectionDetail']['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'

service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)

docsearch = OpenSearchVectorSearch.from_documents(
    docs,
    bedrock_embeddings,
    opensearch_url=host,
    http_auth=auth,
    timeout = 100,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    index_name=index_name,
    engine="faiss",
    bulk_size=1000  # number of documents to be indexed in a single bulk indexing request
)



dynamodb = boto3.resource("dynamodb")

try:
    table = dynamodb.create_table(
        TableName="SessionTable",
        KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    print("Table created successfully.")
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == 'ResourceInUseException':
        print("Table already exists.")
    else:
        # Handle other exceptions or errors
        print("Unexpected error:", e)

