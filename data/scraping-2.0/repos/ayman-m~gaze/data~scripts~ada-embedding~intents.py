import pandas as pd
import openai
import os
from pathlib import Path
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding
import csv

env_path = Path('..') / '.env'
if env_path.exists():
    load_dotenv()
# Retrieve environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

csv_file = open('data/embeddings/intents/ada-basic-intent-embedding.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['name', 'embedding'])

intent_folder = 'data/source/intents/basic'  # Path to the folder containing intent files

# Loop through the intent files
for file_name in os.listdir(intent_folder):
    file_path = os.path.join(intent_folder, file_name)
    intent_name = os.path.splitext(file_name)[0]  # Use file name without extension as intent name

    # Read the intent file
    with open(file_path, 'r') as file:
        samples = file.read().splitlines()

    # Apply the get_embedding function to each sample and write to CSV
    for sample in samples:
        embedding = get_embedding(sample, engine='text-embedding-ada-002')
        csv_writer.writerow([intent_name, embedding])

# Close the CSV file
csv_file.close()

