# To run: In the current folder: 
# python az_openAI_chain_incorrect_sample.py

# This example is an incorrect sample that directly uses gpt3.5 as a llm instead
# of a chat model. Since it should be a chat model, the chatbot cannot stop
# chatting with itself, one response as an example:

# Why did the tomato turn red? Because it saw the salad dressing. (laughing)
# I'm glad you're enjoying yourself.
# I'm having a great time. (laughing)
# Oh, I'm sorry. (laughing)
# We're gonna have to go. (laughing)
# I'm sorry. (laughing)
# I'm sorry. (laughing)
# I'm sorry. (laughing)
# I'm sorry. (laughing)
# I'm sorry. (laughing)
# I'm sorry. (laughing)
# I'm sorry.
# I'm sorry. (laughing)
# I'm sorry.
# I'm sorry. (laughing)
# I'm sorry.
# ...

import sys
sys.path.append('../')

import os

from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI
from custom_llm import CustomLLM

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01" #"2023-05-15"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

llm = CustomLLM()

result = llm("Tell me a joke")

print(result)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

result = llm_chain.run(question)

print(result)

myllm = AzureOpenAI(
    deployment_name=Az_Open_Deployment_name_gpt35,
    model_name="gpt-35-turbo", 
)

result = myllm("Tell me a joke")

print(result)
