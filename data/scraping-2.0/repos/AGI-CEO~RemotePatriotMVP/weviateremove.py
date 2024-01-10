from flask import Flask, request
import weaviate
import os
import openai
from uuid import uuid4
from langchain.text_splitter import TokenTextSplitter

# OpenAI API key configuration
OPENAI_API_KEY = ''
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Weaviate keys configuration
auth_config = weaviate.AuthApiKey(api_key="")
client = weaviate.Client(auth_client_secret=auth_config, url="")
# Retrieve objects from the "job_entries" class
client.schema.delete_class('Knowledge')
