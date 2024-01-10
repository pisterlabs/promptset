from typing import List
import pandas as pd
import ray
import requests
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch
import boto3
import time

ray.init(
    runtime_env={"pip": ["langchain", "sentence_transformers", "transformers","opensearch-py"]}
)

# get region
token_url = 'http://169.254.169.254/latest/api/token'
headers = {'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
response = requests.put(token_url, headers=headers)
token = response.text
response = requests.get('http://169.254.169.254/latest/meta-data/placement/availability-zone', headers={'X-aws-ec2-metadata-token': token})
availability_zone = response.text
region = availability_zone[:-1]

model_name = "sentence-transformers/all-mpnet-base-v2"

#Get variables from cloudformation output of RagStack
cfn_client = boto3.client('cloudformation')
stack_name = 'RAGStack'
stack_outputs = cfn_client.describe_stacks(StackName=stack_name)['Stacks'][0]['Outputs']
for output in stack_outputs:
    if output['OutputKey'] == 'opensearchUrl':
        URL = output['OutputValue']
    if output['OutputKey'] == 'bucketName':
        bucket_name = output['OutputValue']

# Data set has one question per row
ds = ray.data.read_text(f"s3://{bucket_name}/oscar/questions/questions.csv")

class Embed:
    def __init__(self):
        # Specify "cuda" to move the model to GPU.
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.transformer = SentenceTransformer(self.model_name, device="cuda")
        self.embedding_hf=HuggingFaceEmbeddings(model_name=model_name)
        self.URL = URL
        self.DOMAIN = 'genai'
        self.client = OpenSearch(self.URL)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.namespace = 'RAG'
        self.vectordb = 'opensearch'

    def put_cloudwatch_metric(self, time_ms):
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'querytime',
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
        embeddings = self.transformer.encode(
            text_batch['text'].tolist(),
            batch_size=100,  # Large batch size to maximize GPU utilization.
            device="cuda",
        ).tolist()

        r = list(zip(text_batch['text'].tolist(), embeddings))
        for i, (t, e) in enumerate(r):
            query = {
                'size': 5,
                'query': {
                    'knn': {
                        'embedding': {
                            'vector': e,
                            'k': 5
                        }
                    }
                }
            }
            try:
                t1 = time.time()
                response = self.client.search(
                    body = query,
                    index = self.DOMAIN,
                    stored_fields="_none_",
                    _source=False,
                    docvalue_fields=["_id"],
                    filter_path=["hits.hits.fields._id"]
                )
                t2 = time.time()
                self.put_cloudwatch_metric((t2-t1) * 1000.0)
            except Exception as e:
                print(f"Error running OpenSearch query: {e}")
            print(f"Got response: {response}")

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
