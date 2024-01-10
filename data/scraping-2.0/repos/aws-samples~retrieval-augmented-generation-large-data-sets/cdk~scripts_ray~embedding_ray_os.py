from typing import List
import pandas as pd
import ray
import requests
import uuid
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch
import time
import os

os.environ["RAY_DATA_STRICT_MODE"]="0"
ray.init(
    runtime_env={"pip": ["langchain", "sentence_transformers", "transformers","opensearch-py"],
                 "env_vars":{"RAY_DATA_STRICT_MODE":"0" }
                 }
)

# get region
token_url = 'http://169.254.169.254/latest/api/token'
headers = {'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
response = requests.put(token_url, headers=headers)
token = response.text
response = requests.get('http://169.254.169.254/latest/meta-data/placement/availability-zone', headers={'X-aws-ec2-metadata-token': token})
availability_zone = response.text
region = availability_zone[:-1]

#Get variables from cloudformation output of RagStack
cfn_client = boto3.client('cloudformation')
stack_name = 'RAGStack'
stack_outputs = cfn_client.describe_stacks(StackName=stack_name)['Stacks'][0]['Outputs']
for output in stack_outputs:
    if output['OutputKey'] == 'opensearchUrl':
        URL = output['OutputValue']
    if output['OutputKey'] == 'bucketName':
        bucket_name = output['OutputValue']

DOMAIN = 'genai'

# Read data
ds = ray.data.read_parquet(f"s3://{bucket_name}/oscar/parquet_data/")
train, test = ds.train_test_split(test_size=0.1)
ds = test

# Convert to text only
def convert_to_text(batch: pd.DataFrame) -> List[str]:
    return list(batch["content"])
ds = ds.map_batches(convert_to_text)

# Split into chunks
def split_text(page_text: str):
    # Use chunk_size of 1000.
    # We felt that the answer we would be looking for would be 
    # around 200 words, or around 1000 characters.
    # This parameter can be modified based on your documents and use case.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    if page_text and len(page_text) > 0:
        split_text: List[str] = text_splitter.split_text(page_text)
        split_text = [text.replace("\n", " ") for text in split_text]
        return split_text
    else:
        return []
ds = ds.flat_map(split_text)

# Create embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"

class Embed:
    def __init__(self):
        # Specify "cuda" to move the model to GPU.
        self.transformer = SentenceTransformer(model_name, device="cuda")
        self.embedding_hf=HuggingFaceEmbeddings(model_name=model_name)
        self.URL = URL
        self.DOMAIN = DOMAIN
        self.client = OpenSearch(self.URL)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.namespace = 'RAG'
        self.vectordb = 'opensearch'
        os.environ["RAY_DATA_STRICT_MODE"]="0"

    def put_cloudwatch_metric(self, time_ms):
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'ingesttime',
                        'Dimensions': [
                            {
                                'Name': 'vectordb',
                                'Value': self.vectordb
                            }
                        ],
                        'Value': time_ms,
                        'Unit': 'Milliseconds'
                    }
                ]
            )
        except Exception as e:
            print(f"Error writing to CloudWatch: {e}")

    def __call__(self, text_batch: List[str]):
        # We manually encode using sentence_transformer since LangChain
        # HuggingfaceEmbeddings does not support specifying a batch size yet.
        try:
            embeddings = self.transformer.encode(
                text_batch,
                batch_size=100,  # Large batch size to maximize GPU utilization.
                device="cuda",
            ).tolist()

            requests = []
            r = list(zip(text_batch, embeddings))
            ids = []
            for i, (t, e) in enumerate(r):
                _id = str(uuid.uuid4())
                request = {
                    "_op_type": "index",
                    "_index": self.DOMAIN,
                    "embedding": e,
                    "passage": t,
                    "doc_id": _id,
                    "_id": _id
                }
                requests.append(request)
                ids.append(_id)

            t1 = time.time()
            bulk(self.client, requests)
            t2 = time.time()
            self.put_cloudwatch_metric((t2-t1) * 1000.0)
        except Exception as e:
            print(f"Error ingesting to OpenSearch: {e}")

        return r

    


ds = ds.map_batches(
    Embed,
    # Large batch size to maximize GPU utilization.
    # Too large a batch size may result in GPU running out of memory.
    # If the chunk size is increased, then decrease batch size.
    # If the chunk size is decreased, then increase batch size.
    batch_size=100,  # Large batch size to maximize GPU utilization.
    compute=ray.data.ActorPoolStrategy(min_size=20, max_size=20),  # I have 20 GPUs in my cluster
    num_gpus=1,  # 1 GPU for each actor.
)

ds.count()


#finally we want to optimize the OS index to reduce the number of segments.
opensearch_client = OpenSearch(URL)

# Do the force_merge to reduce the number of segments.
def force_merge(os_client):
    i = 1
    while i <= 20:
        try:
            print(f"Force Merge iteration {i}...")
            i = i + 1
            os_client.indices.forcemerge(index=DOMAIN, max_num_segments=1, request_timeout=20000)
            # ensuring the force merge is completed
            break
        except Exception as e:
            print("Waiting for Force Merge to complete")
            time.sleep(300)
            print(f"Running force again due to error..... {e}")

force_merge(opensearch_client)

print(f"Refreshing the index.. {DOMAIN}")
opensearch_client.indices.refresh(index=DOMAIN)