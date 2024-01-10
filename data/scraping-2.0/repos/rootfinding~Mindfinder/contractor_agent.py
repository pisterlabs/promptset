import openai

import os
#load from json .creds/PINECONE_API
import json
with open('.creds/PINECONE_API') as f:
    creds = json.load(f)
    PINECONE_API_KEY = creds['PINECONE_API_KEY']
    PINECONE_ENVIRONMENT = creds['PINECONE_ENVIRONMENT']
    OPENAI_API_KEY = creds['OPENAI_API_KEY']

openai.api_key = OPENAI_API_KEY



