import openai
import pandas as pd
import json
import time
import os
import sys
import chromadb
import openai
import requests
import pandas as pd
import numpy as np
import config
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
from typing import List
from langchain.docstore.document import Document
from langchain.retrievers import ChatGPTPluginRetriever



api_key = config.api_key
openai.api_key = api_key
original_data = pd.read_csv("train.csv")

# original_data[:399].to_json("data.jsonl", orient="records", lines=True)
# original_data[399:].to_json("test.jsonl", orient="records", lines=True)
# positive_examples = original_data[original_data["STATUS"] == 1].head(10)
# negative_examples = original_data[original_data["STATUS"] == 2].head(11)
# selected_examples = pd.concat([positive_examples, negative_examples])
# original_data.to_csv("selected_examples.csv", index=False)

original_data[:399].to_csv("train_resample.csv", index=False)
original_data[399:].to_csv("test_resample.csv", index=False)
train_loader = CSVLoader("./train_resample.csv", source_column="STATUS")
test_loader = CSVLoader("./test_resample.csv", source_column="STATUS")

r_train_data = []
r_test_data = []


def train_data_write_jsonl(path: str, documents: List[Document]) -> None:
    lookup = {"2": "negative", "1": "positive"}
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "STATUS":
                key, value = k.split(": ")
                data_dict[key] = value
        r_train_data.append([str(data_dict), lookup[d.metadata["source"]]])
    # print(r_train_data[0])
    messages = []
    for line in r_train_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies liver diseases"}, {"role": "user", "content":line[0]}, {"role": "assistant", "content": str(line[1])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


def test_data_write_jsonl(path: str, documents: List[Document]) -> None:
    lookup = {"2": "negative", "1": "positive"}
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "STATUS":
                key, value = k.split(": ")
                data_dict[key] = value
        r_test_data.append([str(data_dict)])
    # print(r_train_data[0])
    messages = []
    for line in r_test_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies liver diseases"}, {"role": "user", "content": "Please classify the following patient data based on the provided training data:"}, {"role": "assistant", "content": str(line[0])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


train_data_write_jsonl("foo_train.jsonl", train_loader.load())
test_data_write_jsonl("foo_test.jsonl", test_loader.load())

lookup = {"2": "negative", "1": "positive"}
jsonl_data = []

training_file = openai.File.create(
  file=open("foo_train.jsonl", "r"),
  purpose='fine-tune'
)
status_good = ["uploaded", "processed", "pending"]
status_bad = ["error", "deleting", "deleted"]

while training_file.status not in status_good:
#   time.sleep(5) 
    print(training_file.status)
print("File processed")

job = openai.FineTuningJob.create(
  training_file=training_file.id,
  model="gpt-3.5-turbo-0613",
  suffix = "LVC-Model",
#   n_epochs=2,
)
# print(openai.FineTuningJob.list(limit=10))
print(job)
# print(openai.FineTuningJob.retrieve(job.id))
# print(openai.FineTuningJob.list_events(id=job.id, limit=10))
