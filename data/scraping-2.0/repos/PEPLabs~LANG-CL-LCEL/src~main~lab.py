from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import HuggingFaceEndpoint

import os

from langchain_community.chat_models import ChatHuggingFace

# TODO: Complete this prompt to ask the model for general information on a {topic}:
prompt_template = "{topic}"
prompt = ChatPromptTemplate.from_template(prompt_template)

llm = HuggingFaceEndpoint(
    endpoint_url=os.environ['LLM_ENDPOINT'],
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 1024
    }
)

chat_model = ChatHuggingFace(llm=llm)

# Use a simple output parser that converts output to a string
output_parser = StrOutputParser()

# TODO: Create/return a chain using the prompt, chat_model, and output_parser
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