import numpy as np
import openai
from openai import OpenAI
import os
import pandas as pd
import pickle
import json
from pathlib import Path

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Authenticate with OpenAI API
with open("apiKeys.txt", "r") as temp:
    apiKey = temp.read()
client = OpenAI(api_key=apiKey)


def gpt4(question, tokens=500):
    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="gpt-4", max_tokens=tokens, temperature=0, messages=messages
    )

    # Extract the content
    content = response.choices[0].message.content

    # Split the content into text and code
    text_parts = []
    code_parts = []
    in_code_block = False

    for line in content.split("\n"):
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_parts.append(line)
        else:
            text_parts.append(line)

    # Print the text parts
    for line in text_parts:
        print(line)

    # Print a separator
    print("\n" + "-" * 50 + "\n")

    # Print the code parts
    for line in code_parts:
        print(line)
    return content


# Generate the Question
with open("jsonl_files/combined.jsonl", "r") as f:
    temp = list(f)
content = []
for json_str in temp:
    result = json.loads(json_str)
    content.append(result)

for file in os.listdir("md_files"):
    p_list = []
    p_holder = []
    for item in content:
        if sum(len(i) for i in p_holder) >= 6000:
            p_list.append("\n".join(p_holder))
            p_holder = []
        if item["filename"] == file:
            p_holder.append(f'Question: {item["prompt"]}, Result: {item["completion"]}')
    p_list.append("\n".join(p_holder))

    results = []
    for p_context in p_list:
        prompt = f""" Consider the following information. From it, devise a list of 10 Questions and Answers, based solely on the data presented: {p_context}"""
        results.append(gpt4(prompt, 2000))

    with open(os.path.join("qa_output", str(Path(file).stem) + ".txt"), "w") as f:
        for i in results:
            f.write(i)
