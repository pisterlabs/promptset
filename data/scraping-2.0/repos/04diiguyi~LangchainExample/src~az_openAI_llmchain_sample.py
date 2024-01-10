# To run: In the current folder: 
# python az_openAI_llmchain_sample.py

# This example is a sample that uses OpenAI gpt3 as a llm, and then create
# a LLMchain using this llm
# Example response
# Justin Beiber was born in 1994, 
# so the NFL team that won the Super Bowl that year was the Dallas Cowboys.

import os

from langchain.llms import AzureOpenAI
from langchain import PromptTemplate, LLMChain

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt3

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

llm = AzureOpenAI(
    deployment_name=Az_Open_Deployment_name_gpt3,
    model_name="text-davinci-003", 
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

result = llm_chain.run(question)

print(result)
