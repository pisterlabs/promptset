from openai.embeddings_utils import get_embedding
import openai
import json
import os
from tqdm import tqdm

openai.api_key = "yourAPI_KEY"


def append_to_json_file(file_path, new_objects):
    # Read the existing data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Append new objects to the existing data
    data.append(new_objects)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def split_text_into_chunks(file_path):
    chunks = []

    with open(file_path, 'r') as file:
        text = file.read().replace('\n', ' ')  # Read the file and replace newlines with spaces

    current_chunk = ""
    words = text.split()  # Split the text into words

    for word in words:
        if len(current_chunk) + len(word) <= 2000:
            print("------")
            current_chunk += word + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + ' '

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def list_file(path, text):
    for f in os.listdir(path):
        if os.path.isdir(path + "/" + f):
            text += f"{f} folder have files: "
            text = list_file(path + "/" + f, text)
        else:
            if "html" not in f:
                file = open(path + "/" + f, "r")
                content = file.read()
                content = content.replace("\n", " ")
            else:
                content = "something"
            if not content or content == " ":
                text += f"content of {f} file is empty."
            else:
                text += f"content of {f} file is {content} "
    return text


data_path = "data"
folders = os.listdir(data_path)
f = open("data.txt", "w")
for folder in folders:
    text = f"{folder} include:  "
    if os.path.isdir(data_path + "/" + folder):
        text = list_file(data_path + "/" + folder, text)
    for i in range(10):
        text = text.replace("  ", " ")
    f.write(text)
    f.write("\n")

chunks = split_text_into_chunks('data.txt')
print(len(chunks))
#
for i in tqdm(chunks):
    embd = get_embedding(i, engine='text-embedding-ada-002')
    obj = {
        'chunk_text': i,
        'embeddings': embd
    }

    append_to_json_file('data.json', obj)
