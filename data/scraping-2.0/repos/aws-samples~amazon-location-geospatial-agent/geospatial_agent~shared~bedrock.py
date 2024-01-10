import boto3
from botocore.client import BaseClient
from botocore.config import Config
from langchain.llms import Bedrock

import os


def get_claude_v2(max_tokens_to_sample=8100, temperature=0.001):
    client = get_bedrock_client()
    llm = Bedrock(model_id="anthropic.claude-v2",
                  client=client,
                  model_kwargs={
                      "max_tokens_to_sample": max_tokens_to_sample,
                      "temperature": temperature
                  })
    return llm


def get_bedrock_client() -> BaseClient:
    profile = os.environ.get("AWS_PROFILE", None)
    session = boto3.Session(profile_name=profile)
    cfg = Config(retries={'max_attempts': 10, 'mode': 'adaptive'})
    client: BaseClient = session.client("bedrock-runtime", region_name="us-east-1", config=cfg)
    return client
