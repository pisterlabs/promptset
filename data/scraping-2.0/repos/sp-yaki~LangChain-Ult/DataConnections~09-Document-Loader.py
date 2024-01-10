from langchain.document_loaders import CSVLoader

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

loader = CSVLoader('some_data/penguins.csv')
data = loader.load()
print(data[0].page_content)

from langchain.document_loaders import BSHTMLLoader
loader = BSHTMLLoader('some_data/some_website.html')
data = loader.load()
print(data[0].page_content)

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader('some_data/some_report.pdf')
pages = loader.load_and_split()
print(pages[0].page_content)