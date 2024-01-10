from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
import os
import streamlit as st
import boto3
from botocore.config import Config


def get_client(region):
    session = boto3.Session()
    config = Config(retries = {
        "max_attempts":10,
        "mode":"standard"
    })

    bedrock = session.client("bedrock-runtime", region_name=region, config=config)
    return bedrock

def get_llm(region,credentials_profile,model,model_kwargs,streaming=True):
    #client = boto3.client("bedrock", region)
    client = get_client(region)
    llm = Bedrock(credentials_profile_name=None,model_id=model,region_name=region,client=client,model_kwargs=model_kwargs,streaming=streaming)
    return llm

  