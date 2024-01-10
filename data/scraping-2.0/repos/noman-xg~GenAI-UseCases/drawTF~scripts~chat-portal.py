import requests
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import argparse
import pandas as pd
import logging
import sys
import gradio as gr
import re

# Function to send a POST request with text data and retrieve a response
def get_response(args):
    # Define the URL for the post call
    url = "http://127.0.0.1:8082/message"

    # Define the data to be sent as a JSON object
    data = {"text": args, "path": "/home/xgrid/workspace/openAi/files"}

    # Make the post call and store the response
    response = requests.post(url, json=data, verify=False)

    # Print the status code and the content of the response
    print(response.status_code)
    print(response.text)
    return_val = hcl_parser(response.content)
    refined_val = get_value_between(return_val)
    return str(refined_val)

# Function to extract content between triple backticks (```)
def get_value_between(string):
  # Define a regular expression pattern that matches ``` followed by any characters until another ```
  pattern = r"```(.*)```"
  # Use re.search to find the first match of the pattern in the string
  match = re.search(pattern, string)
  # If there is a match, return the group that contains the value between the ```
  if match:
    return match.group(1)
  # Otherwise, return None
  else:
    return None

# Function to parse and format the response content
def hcl_parser(args):
    args = str(args)
    args = args.replace("\\n", "\n")
    args = args.replace("\\", "")
    args = args.replace('"response":', "")
    return args

# Entry point of the script
if __name__ == "__main__":
    # Create a Gradio interface to interact with the function
    iface = gr.Interface(fn=get_response, inputs="text", outputs="text")
    iface.launch(share=True)
