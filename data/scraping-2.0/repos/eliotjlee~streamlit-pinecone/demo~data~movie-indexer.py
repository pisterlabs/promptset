import os
import openai
import pandas as pd
import pinecone
import time

'''
This script reads in the movie dataset and generates embeddings for each movie plot summary using OpenAI's Embeddings API.
The embeddings are then upserted into a Pinecone index (this enables easy semantic similarity search).
'''

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(      
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_REGION'),  
)      
db = pinecone.GRPCIndex('movies')

# Retrieves OpenAI embeddings for a given text
def get_embeddings(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    # if the text is too long, recursively split it in half and get embeddings for each half then find the average
    except Exception as e:
        # find position of nearest period to the halfway mark
        half_length = len(text) // 2
        period_position = text.rfind('.', 0, half_length)  # search for a period before the halfway mark
        if period_position == -1:  # if no period is found before the halfway mark, search after it
            period_position = text.find('.', half_length)
            if period_position == -1:  # if still not found, just split at the halfway mark
                period_position = half_length

        # split text at the nearest period to the halfway mark
        first_half = text[:period_position+1]  # include the period in the first half
        second_half = text[period_position+1:]

        # get embeddings for each half
        first_half_embeddings = get_embeddings(first_half)
        second_half_embeddings = get_embeddings(second_half)

        # return the average of the two halves
        return [(x + y) / 2 for x, y in zip(first_half_embeddings, second_half_embeddings)]



# Open the csv file and iterate through the rows
data = pd.read_csv('wiki_movie_plots_deduped_with_id.csv')
batch = []
for index, row in data.iterrows():
    ID = row['id']
    summary = row['Plot']

    embeddings = get_embeddings(summary)
    batch.append((str(ID), embeddings))

    # Upsert the batch every 20 items; prevents connection throttling
    if len(batch) == 20:
        while True:
            try:
                db.upsert(batch)
                batch = []
                break
            except Exception as e:
                print(f"Encountered an error: {e}. Retrying in 5 seconds.")
                time.sleep(5)

# Upsert any remaining items in the batch
if batch:
    while True:
        try:
            db.upsert(batch)
            break
        except Exception as e:
            print(f"Encountered an error: {e}. Retrying in 5 seconds.")
            time.sleep(5)
