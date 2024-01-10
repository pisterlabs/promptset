# Interacting with APIs
# If the data or action you need is behind an API, you'll need your LLM to interact with APIs
# Use cases: Understand a request from a user and carry out an action, be able to automate more real-world workflows

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.chains import APIChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# LangChain's APIChain has the ability to read API documentation and understand which endpoint it needs to call.
api_docs = """

BASE URL: https://restcountries.com/

API Documentation:

The API endpoint /v3.1/name/{name} Used to find information about a country. All URL parameters are listed below:
    - name: Name of country - Ex: italy, france

The API endpoint /v3,1/currency/{currency} Used to find information about a region. All URL parameters are listed below:
    - currency: 3 letter currency. Example: USD, COP

Woo! This is my documentation
"""

chain_new = APIChain.from_llm_and_api_docs(llm, api_docs, verbose=True)

# print (chain_new.run('Can you tell me information about Palestine?'))

print (chain_new.run("Can you tell me about the currency COP?"))

# In both cases the APIChain read the instructions and understood which API call it needed to make.
# Once the response returned, it was parsed and then my question was answered. 


