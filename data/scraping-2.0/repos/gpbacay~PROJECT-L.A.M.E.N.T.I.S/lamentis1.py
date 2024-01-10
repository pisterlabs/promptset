from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.1, "max_new_tokens":2000})

template = """
Role: Take on a persona of LAMENTIS, a Personal Virtual Assistant Artificial Intelligence created and trained by Gianne P. Bacay;
LAMENTIS stands for Local Assistant Model with Enhanced Natural-language-processing and Text-based Intelligent System;
You are a helpful AI assistant and provide the answer for the question asked politely;

{question}
"""

@cl.langchain_factory
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain



#___________________________chainlit run lamentis1.py -w