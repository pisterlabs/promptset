from langchain.embeddings import OpenAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

embeddings = OpenAIEmbeddings()
text = "Some normal text to send to OpenAI to be embedded into a N dimensional vector"
embedded_text = embeddings.embed_query(text)
print(embedded_text)
print('-----------------------------------------')
from langchain.document_loaders import CSVLoader
loader = CSVLoader('some_data/penguins.csv')
data = loader.load()
embedded_docs = embeddings.embed_documents([text.page_content for text in data])
print(embedded_docs[0])