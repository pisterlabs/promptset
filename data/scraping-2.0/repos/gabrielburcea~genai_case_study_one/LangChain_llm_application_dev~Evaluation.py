# Databricks notebook source
"""LangChain: Evaluation
Outline:
Example generation
Manual evaluation (and debuging)
LLM-assisted evaluation
LangChain evaluation platform"""

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

"""Create our QandA application"""

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

# COMMAND ----------

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

# COMMAND ----------

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# COMMAND ----------

llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# COMMAND ----------

"""Coming up with test datapoints"""

# COMMAND ----------

data[10]

# COMMAND ----------

data[11]

# COMMAND ----------

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# COMMAND ----------

"""LLM-Generated examples"""

# COMMAND ----------

from langchain.evaluation.qa import QAGenerateChain

# COMMAND ----------

example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

# COMMAND ----------

# the warning below can be safely ignored

# COMMAND ----------

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

# COMMAND ----------

new_examples[0]

# COMMAND ----------

data[0]

# COMMAND ----------

examples += new_examples

# COMMAND ----------

qa.run(examples[0]["query"])

# COMMAND ----------

import langchain
langchain.debug = True

# COMMAND ----------

qa.run(examples[0]["query"])

# COMMAND ----------

# Turn off the debug mode
langchain.debug = False

# COMMAND ----------

"""LLM assisted evaluation¶"""

# COMMAND ----------

predictions = qa.apply(examples)

# COMMAND ----------

from langchain.evaluation.qa import QAEvalChain

# COMMAND ----------

llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)

# COMMAND ----------

graded_outputs = eval_chain.evaluate(examples, predictions)

# COMMAND ----------

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()

# COMMAND ----------

graded_outputs[0]

# COMMAND ----------

"""LangChain evaluation platform
The LangChain evaluation platform, LangChain Plus, can be accessed here https://www.langchain.plus/. Use the invite code lang_learners_2023"""
