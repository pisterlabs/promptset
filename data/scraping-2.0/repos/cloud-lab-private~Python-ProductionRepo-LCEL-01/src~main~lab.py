from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnablePassthrough

import os

# Set Up API access via environment variables:
api_key = os.environ['OPENAI_API_KEY']
base_url = os.environ['OPENAI_API_BASE']
version = os.environ['OPENAI_API_VERSION']

# TODO: Complete this prompt to ask the model for general information on a {topic}:
prompt_template = "{topic}"
prompt = ChatPromptTemplate.from_template(prompt_template)

# Create a model:
model = AzureChatOpenAI(openai_api_version="2023-05-15")

# Use a simple output parser that converts output to a string
output_parser = StrOutputParser()

# TODO: Create/return a chain using the prompt, model, and output_parser
# Make sure you use LCEL to achieve this. 
# Hint: The function body can be as short as a single line
def get_basic_chain():
    chain = None
    return chain

# Using the chain created in basic_chain, invoke the chain with a topic.
# PLEASE DO NOT edit this function
def basic_chain_invoke(topic):
    chain = get_basic_chain()
    try:
        response = chain.invoke({"topic": topic})
    except Exception as e:
        return "Something went wrong: {}".format(e)
    return response
