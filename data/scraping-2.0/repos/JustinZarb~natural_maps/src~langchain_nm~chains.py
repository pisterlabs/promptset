import os
import json
import pandas as pd
import requests
from time import gmtime, strftime
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import (
    TransformChain,
    LLMChain,
    SimpleSequentialChain,
    SequentialChain,
)

api_key = os.getenv("OPENAI_KEY")

llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo-0613")


# Here we have a sequential chain, it contacts overpass and it consist of three steps.

# This chain translates the users's message in an overpass QL query
# (without executing the query)
chain_to_overpass_prompt = PromptTemplate(
    input_variables=["user_text_input"],
    template=""""Turn the user's message into an overpass QL query.
            Example prompt: "Find bike parking near tech parks in Kreuzberg, Berlin.":\n\n {user_text_input}""",
)

chain_to_overpass = LLMChain(
    llm=llm, prompt=chain_to_overpass_prompt, output_key="ql_query"
)


# Function that will perform the actual overpass query
def overpass_query(ql_query):
    """Run an overpass query"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": ql_query})
    if response.content:
        try:
            data = response.json()
        except:
            data = {"error": str(response)}
    else:
        print("Empty response from Overpass API")
        data = {
            "warning": "received an empty response from Overpass API. Tell the user."
        }
    data_str = json.dumps(data)

    return data_str


# Apparently, TransformChain wants functions acting on dictionaries
# You can't directly use general functions
def perform_op_query_func(inputs: dict) -> dict:
    query_input = inputs["ql_query"]
    op_answer = overpass_query(query_input)
    return {"overpass_answer": op_answer}


# This chain performs the actual overpass query
# json file as answer
perform_op_query_chain = TransformChain(
    input_variables=["ql_query"],
    output_variables=["overpass_answer"],
    transform=perform_op_query_func,
)

# This chain calls again the llm  with the same  user message as before,
# but this time accompanied by the json file as context.
chain_to_user_prompt = PromptTemplate(
    input_variables=["overpass_answer", "user_text_input"],
    template=""""Answer the user's message {user_text_input} based on the result of an overpass QL query contained in {overpass_answer}.""",
)

chain_to_user = LLMChain(llm=llm, prompt=chain_to_user_prompt)

# Here is the definition of the whole sequential chain
overpass_sequential_chain = SequentialChain(
    chains=[chain_to_overpass, perform_op_query_chain, chain_to_user],
    input_variables=["user_text_input"],
)
