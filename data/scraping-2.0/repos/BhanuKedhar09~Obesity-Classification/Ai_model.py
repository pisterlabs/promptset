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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


openai.api_key = config.api_key

columns = {"Gender": "Gender",
           "Age": "Age",
           "Height": "Height",
           "Weight": "Weight",
           "family_history_with_overweight": "family_history_with_overweight",
           "FAVC": "Frequent consumption of high caloric food",
           "FCVC": "Frequency of consumption of vegetables",
           "NCP": "Number of main meals",
           "CAEC": "Consumption of food between meals",
           "SMOKE": "SMOKE",
           "CH2O": "Consumption of water daily",
           "SCC": "Calories consumption monitoring",
           "FAF": "Physical activity frequency",
           "TUE": "Time using technology devices",
           "CALC": "Consumption of alcohol",
           "MTRANS": "Transportation used",
           "Category_type": "Category_type"}

original_data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

category_mapping = {
    'Obesity_Type_I': 'obesity',
    'Obesity_Type_II': 'obesity',
    'Obesity_Type_III': 'obesity',
    'Overweight_Level_I': 'overweight',
    'Overweight_Level_II': 'overweight',
    'Normal_Weight': 'normal_weight',
    'Insufficient_Weight': 'underweight'
}

encoding = {
    "normal_weight" : 0,
    "obesity" : 1,
    "overweight" : 2,
    "underweight" : 3
}

original_data['Category_type'] = (
    original_data['NObeyesdad'].map(category_mapping))
# original_data["category_encoding"] = original_data["Category_type"].map(encoding)

original_data.drop("NObeyesdad", axis=1, inplace=True)

data_80_percent, data_20_percent = train_test_split(
    original_data, test_size=0.2, random_state=42)

train_df, validate_df = train_test_split(
    data_80_percent, test_size=0.3, random_state=42)
test_df = data_20_percent



print("Train data shape: ", train_df.shape)
print("Validate data shape: ", validate_df.shape)
print("Test data shape: ", test_df.shape)
print("-" * 150)
train_df.to_csv('train_data.csv', index=False)
validate_df.to_csv('validate_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)


r_train_data = []
r_test_data = []
r_validate_data = []


def train_data_write_jsonl(path: str, documents: List[Document]) -> None:
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "Category_type":
                key, value = k.split(": ")
                ke = columns[key]
                # print(ke)
                data_dict[ke] = value
        r_train_data.append([str(data_dict), d.metadata["source"]])
    messages = []
    for line in r_train_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies obesity levels"}, {
            "role": "user", "content": line[0]}, {"role": "assistant", "content": str(line[1])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


def validate_data_write_jsonl(path: str, documents: List[Document]) -> None:
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "Category_type":
                key, value = k.split(": ")
                ke = columns[key]
                # print(ke)
                data_dict[ke] = value
        r_validate_data.append([str(data_dict), d.metadata["source"]])
    messages = []
    for line in r_validate_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies obesity levels"}, {
            "role": "user", "content": line[0]}, {"role": "assistant", "content": str(line[1])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


def test_data_write_jsonl(path: str, documents: List[Document]) -> None:
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "Category_type":
                key, value = k.split(": ")
                ke = columns[key]
                # print(ke)
                data_dict[ke] = value
        r_test_data.append([str(data_dict).replace(
            "}", "").replace("{", "").replace("'", ""), d.metadata["source"]])
    messages = []
    for line in r_test_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies obesity levels only based on training data"}, {
            "role": "user", "content": "Please classify the following patient data as normal_weight/obesity/overweight/underweight only based on the provided training data:"}, {"role": "user", "content": str(line[0])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


# print(original_data.columns)
train_loader = CSVLoader("./train_data.csv", source_column="Category_type")
validate_loader = CSVLoader("./validate_data.csv",
                            source_column="Category_type")
test_loader = CSVLoader("./test_data.csv", source_column="Category_type")

train_data_write_jsonl("obesity_train.jsonl", train_loader.load())
validate_data_write_jsonl("obesity_validate.jsonl", validate_loader.load())
test_data_write_jsonl("obesity_test.jsonl", test_loader.load())


status_good = ["uploaded", "processed", "pending", "succeeded"]
status_bad = ["error", "deleting", "deleted"]


all_uploded_files = openai.files.list()
print("files in the repo: ", all_uploded_files.data)
print("-" * 150)


for file in all_uploded_files.data:
    try:
        openai.files.delete(file.id)
    except Exception as e:
        print(f"Failed to delete file {file}: {e}")


training_file = openai.files.create(
    file=open("obesity_train.jsonl", "rb"),
    purpose="fine-tune"
)

while training_file.status not in status_good:
    print(training_file.status)
print("Train File processed")
print("-" * 150)

validation_file = openai.files.create(
    file=open("obesity_validate.jsonl", "rb"),
    purpose='fine-tune'
)

while validation_file.status not in status_good:
    print(validation_file.status)
print("Validation File processed")
print("-" * 150)


current_exsiting_files = (openai.files.list()).data
print("Current Existing files in the openai account:")
print(current_exsiting_files)
print("-" * 150)


all_models = (openai.models.list()).data
models_in_my_org = []
print(all_models)
print("-" * 150)
for model in all_models:
    if model.owned_by == "user-flnu6eg6zogaqvaydrcywn39":
        models_in_my_org.append(model.id)
print("Models in my org: ", models_in_my_org)
print("-"*150)

models_to_be_deleted = list(set(models_in_my_org) - set(config.my_models_list))
print("Models to be deleted", models_to_be_deleted, )
print("-"*150)
for model in models_to_be_deleted:
    try:
        openai.models.delete(model)
    except Exception as e:
        print(f"Failed to delete model {model}: {e}")

print("Available models after Deletion", models_to_be_deleted)
print("-"*150)

if openai.models.retrieve("ft:gpt-3.5-turbo-0613:personal::8SBEArVW") is None:
    obesity_fine_tune_model = openai.fine_tuning.jobs.create(
        training_file=training_file.id,
        validation_file=validation_file.id,
        model="gpt-3.5-turbo-1106",
        hyperparameters= {
            "n_epochs": "auto"
        },
        suffix = "obesity_model",
    )
else:
    obesity_fine_tune_model = openai.models.retrieve(
        "ft:gpt-3.5-turbo-0613:personal::8SBEArVW"
    )
    print("The Obesity Model: ")
    print(obesity_fine_tune_model)

print("-"*150)


predicted_output = []
with open("obesity_test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        messages = data["messages"]
        response = openai.chat.completions.create(
            model="ft:gpt-3.5-turbo-0613:personal::8SBEArVW",
            messages=messages,    
        )
        predicted_output.append(response)


print(predicted_output[0])
print("-"*150)
AI_fine_tuned_pred = []
for response in predicted_output :
    if "normal" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(0)
    elif "overweight" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(2)
    elif "obes" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(1)
    elif "underweight" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(3)
    else:
        AI_fine_tuned_pred.append((response.choices[0]).message.content)

print("AI_fine_tuned_pred: ")
print(AI_fine_tuned_pred)

print("-"*150)


test_df["category_encoding"] = test_df["Category_type"].map(encoding)

print("test_df: ")
print(test_df["category_encoding"].values, len(test_df["category_encoding"].values))
print("-"*150)

print("length of prediction encoded and number of responses back : ", len(AI_fine_tuned_pred), len(predicted_output))
print("-" * 150)

print("Accuracy of the model: ", accuracy_score(test_df["category_encoding"].values, AI_fine_tuned_pred))
print("-" * 150)

print("Confusion Matrix: ", confusion_matrix(test_df["category_encoding"].values, AI_fine_tuned_pred))
print("-" * 150)

print("Classification Report: ", classification_report(test_df["category_encoding"].values, AI_fine_tuned_pred))
print("-" * 150)

