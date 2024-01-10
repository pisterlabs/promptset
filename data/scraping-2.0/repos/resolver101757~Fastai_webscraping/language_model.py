import os
import requests
import json
from openai import OpenAI

# Read the API key from the file
with open('api_key.txt', 'r') as file:
    api_key = file.read().strip()

# This code is for v1 of the openai package: pypi.org/project/openai
client = OpenAI(api_key=api_key)

# The folder containing the files
folder_path = 'output'

# Function to read files in a folder
def read_files_in_folder(folder_path):
    file_contents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # assuming text files
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
    return file_contents

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def summarize(message):
    # Split the message into smaller chunks of 4096 tokens each
    chunks = [message[i:i+4096] for i in range(0, len(message), 4096)]
    # Initialize an empty list to store the summaries
    summaries = []
    # Loop through each chunk and send a request to the Completions API
    for chunk in chunks:
        response = client.completions.create(
            model="gpt-4-1106-preview",  # Replace with an appropriate model name
            prompt=chunk,
            temperature=0,
            max_tokens=8000
        )
        # Extract the summary from the response and append it to the list
        summary = response.choices[0].text
        summaries.append(summary)
    # Join the summaries into a single string and return it
    return " ".join(summaries)



prompt_text = read_text_file('summary_prompt.txt')

# Reading files
file_data = read_files_in_folder(folder_path)

# Prepare data for API call (example: sending one file's contents at a time)
for content in file_data:
    data = content
    response = summarize(data)
    details = response.choices[0].message.content
    print(details)  # handle the details as needed
