from handyllm import OpenAIAPI
from handyllm import utils

import json
from dotenv import load_dotenv, find_dotenv
# load env parameters from file named .env
# API key is read from environment variable OPENAI_API_KEY
# organization is read from environment variable OPENAI_ORGANIZATION
load_dotenv(find_dotenv())

## list all files
response = OpenAIAPI.files_list()
print(json.dumps(response, indent=2))

## upload a file
with open("mydata.jsonl", "rb") as file_bin:
    response = OpenAIAPI.files_upload(
        file=file_bin,
        purpose='fine-tune'
    )
print(json.dumps(response, indent=2))
