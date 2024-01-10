# Databricks notebook source
"""LangChain: Q&A over Documents
An example might be a tool that would allow you to query a product catalog for items of interest."""

# COMMAND ----------

import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# COMMAND ----------

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

# COMMAND ----------

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# COMMAND ----------

from langchain.indexes import VectorstoreIndexCreator

# COMMAND ----------

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# COMMAND ----------

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

# COMMAND ----------

response = index.query(query)

# COMMAND ----------

display(Markdown(response))

# COMMAND ----------

"""Step By Step"""

# COMMAND ----------

from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

# COMMAND ----------

docs = loader.load()

# COMMAND ----------

docs[0]

# COMMAND ----------

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# COMMAND ----------

embed = embeddings.embed_query("Hi my name is Harrison")

# COMMAND ----------

print(len(embed))

# COMMAND ----------

print(embed[:5])

# COMMAND ----------

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

# COMMAND ----------

query = "Please suggest a shirt with sunblocking"

# COMMAND ----------

docs = db.similarity_search(query)

# COMMAND ----------

len(docs)

# COMMAND ----------

docs[0]

# COMMAND ----------

retriever = db.as_retriever()

# COMMAND ----------

llm = ChatOpenAI(temperature = 0.0, model=llm_model)

# COMMAND ----------

qdocs = "".join([docs[i].page_content for i in range(len(docs))])


# COMMAND ----------

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 


# COMMAND ----------

display(Markdown(response))

# COMMAND ----------

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

# COMMAND ----------

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

# COMMAND ----------

response = qa_stuff.run(query)

# COMMAND ----------

display(Markdown(response))

# COMMAND ----------

response = index.query(query, llm=llm)

# COMMAND ----------

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
