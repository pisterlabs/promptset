""" 

Function to get embeddings from OpenAI API and verify the data
Every object in the json file should have an embedding
Every token used has been logged in the TOTAL_TOKENS_USED.txt file


"""

import openai
import os
from dotenv import load_dotenv
import json
import time

import tiktoken
load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = "https://tu-openai-api-management.azure-api.net/OLTATKULL"
openai.api_version = "2023-07-01-preview"


# Function to get embeddings from OpenAI API and log the tokens used

log_file = "TOTAL_TOKENS_USED.txt"


def get_embedding(query):
    try:
        assert isinstance(query, str), "`query` should be a string"
        response = openai.Embedding.create(
            engine="IDS2023_PIKANI_EMBEDDING",
            model="text-embedding-ada-002",
            input=query
        )
        embedding = response.data[0].embedding
        # --------LOGS THE TOKENS--------- #
        with open(log_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('embeddings_tokens_used='):
                total_tokens_used = int(line.split('=')[1])
                break
        tokens = response['usage']['total_tokens']
        total_tokens_used += tokens
        for i, line in enumerate(lines):
            if line.startswith('embeddings_tokens_used='):
                lines[i] = f'embeddings_tokens_used={total_tokens_used}\n'
                break
        with open(log_file, 'w') as f:
            f.writelines(lines)
        # --------------------------------- #
        return embedding, tokens
    except Exception as e:
        print("An error occurred:", e)
        return None


# Fetch embeddings for all courses in the json file

def fetch_embeddings(file):
    total_tokens = 0
    try:
        with open(file, 'r') as f:
            data = json.load(f)

        for obj in data:
            if "embedding" in obj:
                continue
            elif total_tokens <= 145000:
                input_str = json.dumps(obj)
                response = get_embedding(input_str)

                obj["data"] = input_str
                obj["tokens"] = response[1]
                obj["embedding"] = response[0]
                with open(file, 'w') as f:
                    json.dump(data, f, indent=4)
                total_tokens += response[1]
                print(f'{obj["title"]} Done!')
            else:
                print("Reached the limit of tokens, waiting 60 seconds...")
                time.sleep(61)
                total_tokens = 0
    except Exception as e:
        print(f'An error occurred with {obj["title"]}:', e)


def count_embeddings(file):
    with open(file, 'r') as f:
        data = json.load(f)

    num_with_embeddings = 0
    num_without_embeddings = 0

    for obj in data:
        if "embedding" in obj:
            num_with_embeddings += 1
        else:
            num_without_embeddings += 1

    print(
        f"The number of objects with embeddings is {num_with_embeddings}.")
    print(
        f"The number of objects without embeddings is {num_without_embeddings}.")


fetch_embeddings("data/embeddings.json")
# count_embeddings("data/embeddings.json")
