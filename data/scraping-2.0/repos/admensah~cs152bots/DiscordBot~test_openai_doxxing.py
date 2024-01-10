import os
import json
import logging
import re
import requests
import pdb
import heapq
import openai
import pandas as pd
from tqdm import tqdm

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    openai.organization = tokens['openai_org']
    openai.api_key = tokens['openai_api_key']


def eval_text(message):
    conversation = [
        {"role": "system", "content": "You are a content moderation system. Classify each input as either Doxxing, Extortion, Threats, Sexual Harassment, Hate Speech, Bullying, or . Then assign a severity level to it between 1 and 4, 4 being the most severe. The message you return should be in the format 'Type (Severity)' unless its Doxxing then return 'Doxxing (Type of Doxxing)' or 'clean' if it is a normal message. Note: Just because a message has swear words, does not necessarily mean that it should be flagged. If a message is clean you should only reply 'clean'."},
        {"role": "user", "content": message}
        ]
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=conversation,
    max_tokens=10  # Adjust the max tokens based on the desired response length
    )
    # TODO: conversation should be kept track of so GPT-4 has more context and can make better decisions
    # TODO: either here or somewhere else, if its doxxing or something very severe we might want to remove the post
    # otherwise we would just send it to the mod channel with the description
    return [message, response.choices[0].message.content]

dataset_path = "../datasets/doxxing_dataset.csv"
with open(dataset_path) as f:
    df = pd.read_csv(f)
df = df[["text", "doxxing"]]

confusion_mat = [[0, 0], [0, 0]]
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text, label = row

    gpt_response = eval_text(text)[1]
    gpt_label = 0 if "clean" in gpt_response else 1

    confusion_mat[label][gpt_label] += 1

print(confusion_mat)
true_positives = confusion_mat[1][1]
true_negatives = confusion_mat[0][0]
false_positives = confusion_mat[0][1]
false_negatives = confusion_mat[1][0]

total = true_negatives + true_positives + false_negatives + false_positives
accuracy = (true_positives + true_negatives) / total
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

    