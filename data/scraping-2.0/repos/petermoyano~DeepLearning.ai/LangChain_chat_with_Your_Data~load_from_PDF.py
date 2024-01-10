from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
import os
import openai
import sys
sys.path.append('../..')

_ = load_dotenv(find_dotenv())  # Loads environment variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")


loader = PyPDFLoader("docs/ReAct_Synergizing_reasoning_and_acting_in_LLMs.pdf")

pages = loader.load()  # Returns a list of Page objects (document objects)
print(len(pages))
page = pages[8]
print(page.metadata)
