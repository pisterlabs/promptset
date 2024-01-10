from decouple import config
from datetime import datetime
import time
import pandas as pd
import os
import openai
from ratelimit import limits, sleep_and_retry

now = datetime.now()
openai.api_key = str(config('API_OPENAI'))

batch_size = 600

@sleep_and_retry
@limits(calls=60, period=60)
def create_embedding(input_text, max_retries=30):
    num_retries = 0
    while num_retries < max_retries:
        try:
            return openai.Embedding.create(input=input_text, engine='text-embedding-ada-002')['data'][0]['embedding']
        except Exception as e:
            print(f"Error calling OpenAI embeddings API: {e}")
            num_retries += 1
            time.sleep(1)  # wait for 1 second before retrying
    raise Exception("Failed to call OpenAI embeddings API after multiple retries.")

df = pd.read_csv('processed/short.csv', index_col=0)
df.columns = ['text', 'tokens']

print(df.head())

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].copy()
    batch['embeddings'] = batch['text'].apply(create_embedding)
    batch.to_csv('processed/embed/embeddings-batch:' + str(i) + '.csv')
    batch.to_csv('processed/embed history/embeddings-batch:' + str(i) +  ";rows:" + str(len(batch.index)) + ";time:" + now.strftime("%m-%d-%Y %H:%M:%S") + '.csv')
    print(batch.head())

csv_dir = "processed/embed"

# Get a list of all the CSV files in the directory
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Create an empty list to store dataframes
dfs = []

# Loop through each CSV file, read it into a dataframe, and append it to the list
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

# Concatenate all the dataframes in the list into a single dataframe
combined_df = pd.concat(dfs)

# Save the combined dataframe to a new CSV file
combined_df.to_csv("processed/embed/embed-comb.csv")
combined_df.to_csv("processed/embed/embed" + ";time:" + now.strftime("%m-%d-%Y %H:%M:%S") + '.csv')