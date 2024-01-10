import openai
import pinecone
import pathlib
import tiktoken
import sys
import re
import os
from tqdm.auto import tqdm
from math import floor
import mysql.connector
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Pinecone settings
index_name = os.getenv("PINECONE_INDEX_NAME")
upsert_batch_size = 50   # how many embeddings to insert at once in the db

# OpenAI settings
embed_model = "text-embedding-ada-002" # embedding model compatible with gpt3.5
max_tokens_model = 8191                # max tokens accepted by embedding model
encoding_model = "cl100k_base"         # tokenizer compatible with gpt3.5

# https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Convert date to string
def date_converter(o):
    return str(o.strftime("%Y-%m-%d"))

def get_daily_report_data():
    # Connect to database
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        port=3306
    )
    cursor = connection.cursor()
    # Query daily report data
    query = "SELECT dr.id AS daily_report_id, dr.report_date AS report_date, t.content, u.username FROM daily_reports dr JOIN tasks t ON dr.id = t.daily_report_id JOIN users u ON dr.user_id = u.id ORDER BY dr.report_date DESC LIMIT 100"
    cursor.execute(query)

    daily_report_data = cursor.fetchall()

    # Convert data to list of dictionaries
    new_data = []
    for row in daily_report_data:
        daily_report_id, report_date, content, username = row
        new_data.append({
            'id': f"daily_report_{daily_report_id}",
            'text': content,
            'report_date': date_converter(report_date),
            'username': username
        })

    # Close connection
    connection.close()
    return new_data

# Initialize connection to Pinecone 
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, enviroment=env)
index = pinecone.Index(index_name)

new_data = get_daily_report_data()

print(f"Extracted {len(new_data)} from rs database: {os.getenv('DB_NAME')}")
print(f"Example data: {new_data[1:5]}")
# Create embeddings and upsert the vectors to Pinecone
print(f"Creating embeddings and uploading vectors to database")
for i in tqdm(range(0, len(new_data), upsert_batch_size)):
    
    # process source text in batches
    i_end = min(len(new_data), i+upsert_batch_size)
    meta_batch = new_data[i:i_end]
    ids_batch = [x['id'] for x in meta_batch]
    texts = [x['text'] for x in meta_batch]

    # compute embeddings using OpenAI API
    embedding = openai.Embedding.create(input=texts, engine=embed_model)
    embeds = [record['embedding'] for record in embedding['data']]

    # clean metadata before upserting
    meta_batch = [{
        'id': x['id'],
        'text': x['text'],
        'report_date': x['report_date'],
        'username': x['username']
    } for x in meta_batch] 

    # upsert vectors
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    index.upsert(vectors=to_upsert)

# Print final vector count
vector_count = index.describe_index_stats()['total_vector_count']
print(f"Database contains {vector_count} vectors.")
