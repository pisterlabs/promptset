import os
# import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
# from getpass import getpass
from dotenv import load_dotenv



HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
load_dotenv()

model_id = "albert-xlarge-v1"
conv_model = HuggingFaceHub(huggingfacehub_api_token=
                            os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8, "max_new_tokens":150})

template = """You are a story writer AI assistant that completes a story based on the query received as input

{query}
"""

prompt = PromptTemplate(input_variables=["query"], template=template)

chain = LLMChain(llm=conv_model, prompt=prompt, output_key="story")

query = input("Enter a query:")
response = chain({"query":query})
print(response["story"])
