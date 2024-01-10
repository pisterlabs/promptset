#Document loading

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

#PDFs

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("/Users/aaliyahsalia/Desktop/SFBU/6thTrimester/CS589/Week6_HW3/2023Catalog.pdf")
pages = loader.load()

print(len(pages))
page = pages[0]
print(page.page_content[0:500])
print(page.metadata)

################

