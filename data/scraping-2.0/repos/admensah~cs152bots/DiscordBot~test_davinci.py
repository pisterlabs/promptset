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


def eval_text(messages):
    #conversations = [[
    #    {"role": "system", "content": "You are a content moderation system. Classify each input as either Doxxing, Extortion, Threats, Sexual Harassment, Hate Speech, Bullying, or . Then assign a severity level to it between 1 and 4, 4 being the most severe. The message you return should be in the format 'Type (Severity)' unless its Doxxing then return 'Doxxing (Type of Doxxing)' or 'clean' if it is a normal message"},
    #    {"role": "user", "content": message}
    #    ] for message in messages]
    #response = openai.ChatCompletion.create(
    #model="gpt-3.5-turbo",
    #messages=conversations,
    #max_tokens=10  # Adjust the max tokens based on the desired response length
    #)
    prompts = [
        f"You are a content moderation system. Classify each input as either Doxxing, Extortion, Threats, Sexual Harassment, Hate Speech, or Bullying. Then assign a severity level to it between 1 and 4, 4 being the most severe. The message you return should be in the format 'Classification (Severity)' unless its Doxxing then return 'Doxxing (Type of Doxxing)' or 'clean' if it is a normal message. Note: just because a message has swear words, does not mean it should be flagged. If a message is clean, it should only say 'clean'"
        + "\nMessage: {message}"
        for message in messages]
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompts,
    max_tokens=16  # Adjust the max tokens based on the desired response length
    )
    # TODO: conversation should be kept track of so GPT-4 has more context and can make better decisions
    # TODO: either here or somewhere else, if its doxxing or something very severe we might want to remove the post
    # otherwise we would just send it to the mod channel with the description
    
    # Match responses
    gpt_responses = [None] * len(messages)
    for choice in response.choices:
        gpt_responses[choice.index] = choice.text.strip()
    
    return gpt_responses

dataset_path = "../datasets/youtube_parsed_dataset.csv"
with open(dataset_path) as f:
    df = pd.read_csv(f)
df = df[["Text", "oh_label"]]

confusion_mat = [[0, 0], [0, 0]]
batch_size = 20
df_list = list(df.iterrows())
gpt_responses_cum = []
for start in tqdm(range(0, df.shape[0], batch_size)):
    texts = [row[0] for _, row in df_list[start:start+batch_size]]
    labels = [row[1] for _, row in df_list[start:start+batch_size]]
    print(texts)
    print(labels)

    gpt_responses = eval_text(texts)
    gpt_responses_cum.extend(gpt_responses)
    gpt_labels = [0 if "clean" in gpt_response else 1 for gpt_response in gpt_responses]

    for label, gpt_label in zip(labels, gpt_labels):
        confusion_mat[label][gpt_label] += 1

# Save the responses as a json
gpt_responses_cum_json_path = "gpt_responses.json"
gpt_responses_cum_json = json.dumps(gpt_responses_cum)
with open(gpt_responses_cum_json_path, "w") as f:
    f.write(gpt_responses_cum_json)

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