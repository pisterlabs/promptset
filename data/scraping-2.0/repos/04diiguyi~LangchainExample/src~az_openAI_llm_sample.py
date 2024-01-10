# To run: In the current folder: 
# python az_openAI_llm_sample.py

# This example is a sample that uses OpenAI gpt3 as a llm in langchain
# Example response
# Q: What did the fish say when he hit the wall?
# A: Dam!

import os

from langchain.llms import AzureOpenAI

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt3

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01" #"2023-05-15"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name=Az_Open_Deployment_name_gpt3,
    model_name="text-davinci-003", 
)

result = llm("Tell me a joke")

print(result)
