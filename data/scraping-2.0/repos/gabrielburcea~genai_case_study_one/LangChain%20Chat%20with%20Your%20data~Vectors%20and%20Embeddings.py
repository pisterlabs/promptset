# Databricks notebook source
"""Vectorstores and Embeddings¶
Recall the overall workflow for retrieval augmented generation (RAG):"""

# COMMAND ----------

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# COMMAND ----------

"""We just discussed Document Loading and Splitting."""

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# COMMAND ----------

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

# COMMAND ----------

splits = text_splitter.split_documents(docs)

# COMMAND ----------

len(splits)

# COMMAND ----------

"""Embeddings
Let's take our splits and embed them."""

# COMMAND ----------

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

# COMMAND ----------

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

# COMMAND ----------

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

# COMMAND ----------

import numpy as np

# COMMAND ----------

np.dot(embedding1, embedding2)

# COMMAND ----------

np.dot(embedding1, embedding3)

# COMMAND ----------

np.dot(embedding2, embedding3)

# COMMAND ----------

"""Vectorstores"""

# COMMAND ----------

# ! pip install chromadb

# COMMAND ----------

from langchain.vectorstores import Chroma

# COMMAND ----------

persist_directory = 'docs/chroma/'

# COMMAND ----------

!rm -rf ./docs/chroma  # remove old database files if any

# COMMAND ----------

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

# COMMAND ----------

print(vectordb._collection.count())

# COMMAND ----------

"""Similarity Search"""

# COMMAND ----------

question = "is there an email i can ask for help"

# COMMAND ----------

docs = vectordb.similarity_search(question,k=3)

# COMMAND ----------

len(docs)

# COMMAND ----------

docs[0].page_content

# COMMAND ----------

"""Let's save this so we can use it later!"""

# COMMAND ----------

vectordb.persist()

# COMMAND ----------

"""Failure modes
This seems great, and basic similarity search will get you 80% of the way there very easily.

But there are some failure modes that can creep up.

Here are some edge cases that can arise - we'll fix them in the next class."""

# COMMAND ----------

question = "what did they say about matlab?"

# COMMAND ----------

docs = vectordb.similarity_search(question,k=5)

# COMMAND ----------

"""Notice that we're getting duplicate chunks (because of the duplicate MachineLearning-Lecture01.pdf in the index).

Semantic search fetches all similar documents, but does not enforce diversity.

docs[0] and docs[1] are indentical."""

# COMMAND ----------

docs[0]

# COMMAND ----------

docs[1]

# COMMAND ----------

""""We can see a new failure mode.

The question below asks a question about the third lecture, but includes results from other lectures as well."""

# COMMAND ----------

question = "what did they say about regression in the third lecture?"

# COMMAND ----------

docs = vectordb.similarity_search(question,k=5)

# COMMAND ----------

for doc in docs:
    print(doc.metadata)

# COMMAND ----------

print(docs[4].page_content)

# COMMAND ----------

"""Approaches discussed in the next lecture can be used to address both!"""

# COMMAND ----------



# COMMAND ----------


