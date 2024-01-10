# load data
# split data
# create emebeddings and store in vector database
# get prompts for chat and do a similarity search in embeddings

import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['AZURE_OPENAI_KEY']
openai.api_type = "azure"
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
openai.api_version = "2023-05-15"


modelname = "aazeem-chat-gpt35"

print("openai base url", openai.api_base)
print("openai api version", openai.api_version)

response = openai.Completion.create(engine=modelname,
                                 temperature=0.0,
                                    max_tokens=20,
                                 prompt="Tell me the capital city of France",
                                 completion_tokens=10
)

print(response)