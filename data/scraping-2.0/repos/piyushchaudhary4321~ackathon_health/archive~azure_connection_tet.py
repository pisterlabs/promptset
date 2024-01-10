import openai
import os
import pandas as pd
import time
from openai import OpenAI

# openai.api_key = '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'


api_key = "1e312aee735446d196382ebd49213641"
api_version = "2023-07-01-preview"
api_type = "azure"
api_base = "https://chatbot-acko.openai.azure.com/"

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="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",
# )

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # api_key="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",
    api_key = "1e312aee735446d196382ebd49213641",
    api_version = "2023-07-01-preview",
    api_type = "azure",
    api_base = "https://chatbot-acko.openai.azure.com/"

)


def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

prompt = "Is Bangalore Crowded"

print(chat_gpt('Is Bangalore Crowded'))

#NEw version

import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")
from langchain.chains import LLMChain
import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import warnings
from openai import AzureOpenAI
warnings.filterwarnings("ignore")

openai.api_key = "1e312aee735446d196382ebd49213641"
# openai.api_version = "2023-07-01-preview"
# openai.api_type = "azure"
openai.api_base = "https://ackocarelife.openai.azure.com/"


client = AzureOpenAI(
    api_key="1e312aee735446d196382ebd49213641",
    api_version="2023-10-01-preview",
    azure_endpoint = "https://ackocarelife.openai.azure.com/"
    )

  
deployment_name='REPLACE_WITH_YOUR_DEPLOYMENT_NAME' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    
# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = client.completions.create(model='ackocarelife', prompt=start_phrase, max_tokens=10)
print(response.choices[0].text)

llm = ChatOpenAI(temperature=0.1, openai_api_key=openai.api_key, engine="ackocarelife")

TEMPLATE_PARSER_1 = """ 
As a sophisticated language model, your task is to analyze the given text and extract key information into a structured dictionary format. 
Pay attention to identifying specific details such as invoice_number, invoice_date, registration_number, labour_code, and other relevant 
variables as given in the examples below. 
Ensure accuracy and relevance in the extraction process.

For your reference, here are a few examples demonstrating how to approach this task:

New Text for Extraction:
[{text}]

"""

def response_parser(text):
    PROMPT_PARSER = PromptTemplate(input_variables=["text"], template=TEMPLATE_PARSER_1,
                                    output_key="output")
    chain_process = LLMChain(llm=llm, prompt=PROMPT_PARSER, output_key='output')
    return chain_process.run({'text': text})

# chain_process= LLMChain(llm=llm, prompt='Is Bangalore Crowded')
# chain_process.run()

temp = response_parser('')

print(temp)

