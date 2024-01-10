import openai
import os
import tiktoken
from util import *
import pandas as pd

openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_qa(entity_id, entity_name):
    print(f"entity_id, entity_name: {entity_id}, {entity_name}")

    with open(f"../data/wikipedia/{entity_id}.txt") as f:
        text = f.read()
    clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';',' ')

    template_messages = [
        {"role": "system", "content": "You are a helpful annotator of sentences."},
        {"role": "user", "content": "Please convert entity name to a mask token.\nThis is examples when the entity name is Chihuahua (dog):\nBefore: What is the origin of the Chihuahua breed of dog?\nAfter: What is the origin of the object breed of dog?\n\n"},
    ]
    template_content = f"Wikipedia article of {entity_name}:\n <document>"

    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = create_chunks(clean_text,4000,tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    results = []
    MODEL = "gpt-3.5-turbo"

    for chunk in text_chunks:
        results.append(extract_chunk(chunk, template_messages, template_content, MODEL))

    output = ""
    for result in results:
        output += result + "\n\n"
    output = output.rstrip()
    print(output)
    print()

    with open(f"../data/gpt_3_output/{entity_id}.txt", 'w') as f:
        f.write(output)

def main():
    with open("../data/entity.csv") as f:
        df = pd.read_csv(f)
    for entity_id, entity_name in zip(df["id"], df["name"]):
        generate_qa(entity_id, entity_name)

if __name__ == "__main__":
    main()

