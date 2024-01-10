import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # This is to load your API key from an .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

#! pip install pypdf
from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("docs/Assura-Basis_CGA_LAMal_2024_F.pdf")

pages = loader.load()

print(len(pages))

page = pages[0]

print(page.page_content[:500])
print(page.metadata)
