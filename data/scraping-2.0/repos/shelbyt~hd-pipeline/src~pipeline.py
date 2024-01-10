# Input: CSV file
# Operations:
# 1. Run each row through gpt to create a sentence (OAI GPT4 API)
# 2. Convert each sentence to an embedding vector (1536) (OAI ADA-EMBED API)
# 3. Batch Upsert the embedding vector into Pinecone using the index, vector, and column info as metadata (Pinecone API)

import os
import pinecone
import csv
from openai import OpenAI
from dotenv import load_dotenv
import argparse

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize APIs
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
# Function to generate sentence using GPT-4
def generate_sentence(input_row, columns=None):
    model = "gpt-4-1106-preview"
    prompt = f"convert this row from a dataset into a 100 word concise but descriptive paragraph with all the  technical specs that I can convert into an embedding. Here are the columns for the dataset. Please ensure data from each available column must included: {columns} -> {input_row}"
    response = client.chat.completions.create(
    model=model,
        messages=[
        {"role": "system", "content": "You are an advanced high iq human who follows instructions exactly."},
        {"role": "user", "content": prompt},
    ],
    max_tokens=3000)
    return {"prompt": prompt, "response": response.choices[0].message.content, "model": model}

# Function to convert sentence to embedding
def convert_to_embedding(sentence):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=sentence
    ) 
    return response.data[0].embedding

# Function to upsert into Pinecone
def upsert_to_pinecone(id, vector, metadata):
    index = pinecone.Index('home-depot')
    index.upsert ([(id, vector, metadata)])


def main():
    pinecone.describe_index("home-depot")
    index = pinecone.Index("home-depot")
    parser = argparse.ArgumentParser(description='Process a CSV file.')
    parser.add_argument('csvfile', type=str, help='The CSV file to process')
    args = parser.parse_args()

    with open(args.csvfile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = " ".join(reader.fieldnames)  # Get headers as a space-separated sentence
        for i,row in enumerate(reader):
            row_str = " ".join([str(val) for val in row.values()])
            generated_sentence_info = generate_sentence(row_str, columns)
            embedding_vector = convert_to_embedding(generated_sentence_info['response'])

            metadata = {key: row[key] for key in row}
            metadata['generated_sentence'] = str(generated_sentence_info['response'])
            metadata['prompt'] = str(generated_sentence_info['prompt'])
            metadata['model'] = str(generated_sentence_info['model'])

            upsert_to_pinecone(row['id'], embedding_vector, metadata)

            print(i, row_str, generated_sentence_info['prompt'], generated_sentence_info['model'], generated_sentence_info['response'])

if __name__ == "__main__":
    main()



