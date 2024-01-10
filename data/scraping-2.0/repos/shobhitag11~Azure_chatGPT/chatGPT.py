import os
import openai
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA

ENDPOINT = "https://<project_id>.openai.azure.com"
API_KEY = ""
DEPLOYMENT_NAME = "text-davinci-003"#gpt-35-turbo-16k, gpt-35-turbo, gpt-4-32k, gpt-4
API_TYPE = "azure"
API_VERSION = "2022-12-01"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_VERSION"] = API_VERSION

openai.api_type = API_TYPE
openai.api_version = API_VERSION
openai.api_base = ENDPOINT
openai.api_key = API_KEY

qa = RetrievalQA.from_chain_type(llm=AzureOpenAI(temperature=0.1, deployment_name=DEPLOYMENT_NAME), chain_type="stuff")

query = "How to Manage Work-life balance at corporate?"
print(qa.run(query))
