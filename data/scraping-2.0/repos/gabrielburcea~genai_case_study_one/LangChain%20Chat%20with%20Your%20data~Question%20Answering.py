# Databricks notebook source
"""Question Answering¶
Overview
Recall the overall workflow for retrieval augmented generation (RAG):
We discussed Document Loading and Splitting as well as Storage and Retrieval.

Let's load our vectorDB.

"""




# COMMAND ----------

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# COMMAND ----------

"""The code below was added to assign the openai LLM version filmed until it is deprecated, currently in Sept 2023. LLM responses can often vary, but the responses may be significantly different when using a different model version."""

# COMMAND ----------

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

# COMMAND ----------

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# COMMAND ----------

print(vectordb._collection.count())

# COMMAND ----------

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

# COMMAND ----------

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# COMMAND ----------

"""RetrievalQA chain"""

# COMMAND ----------

from langchain.chains import RetrievalQA

# COMMAND ----------

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

# COMMAND ----------

result = qa_chain({"query": question})

# COMMAND ----------

result["result"]

# COMMAND ----------

"""Prompt"""

# COMMAND ----------

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


# COMMAND ----------

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# COMMAND ----------

question = "Is probability a class topic?"

# COMMAND ----------

result = qa_chain({"query": question})

# COMMAND ----------

result["result"]

# COMMAND ----------

result["source_documents"][0]

# COMMAND ----------

"""RetrievalQA chain types"""

# COMMAND ----------

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

# COMMAND ----------

result = qa_chain_mr({"query": question})

# COMMAND ----------

result["result"]

# COMMAND ----------

"""If you wish to experiment on the LangChain plus platform:

Go to langchain plus platform and sign up
Create an API key from your account's settings
Use this API key in the code below
uncomment the code
Note, the endpoint in the video differs from the one below. Use the one below."""

# COMMAND ----------

#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key

# COMMAND ----------

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]

# COMMAND ----------

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]

# COMMAND ----------

"""RetrievalQA limitations
QA fails to preserve conversational history."""

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


