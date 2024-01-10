import openai
import pandas as pd
import time
import json
import os
import pinecone
from tqdm.auto import tqdm

# set up the API key
openai.api_key = os.environ.get('OPENAI_API_KEY')  

# import json as dict (generate file by running clean_medical_dialogues.py)
with open(os.path.join(os.path.dirname(__file__), "medical_dialogues_cleaned.json")) as json_file:
    data = json.load(json_file)

def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    retry_count = 0
    while retry_count < 3:
        try:
            result = openai.Embedding.create(
              model=model,
              input=text
            )
            return result
        except Exception as e:
            print(e)
            retry_count += 1
            if retry_count == 3:
                print("Skipping batch with error:", e)
            print("Caught exception, retrying in 15 seconds...")
            time.sleep(15)

# set index name
index_name = 'medical-dialog-embeddings'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key = os.environ.get('PINECONE_API_KEY'),
    environment = os.environ.get('PINECONE_ENVIRONMENT')  
)
# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=len(1536))
# connect to index
index = pinecone.Index(index_name)

# convert to pandas dataframe
df = pd.DataFrame(data)
count = 0  # we'll use the count to create unique IDs
batch_size = 50  # process everything in batches of 50
for i in tqdm(range(0, len(df), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(df))
    # get batch of lines and IDs
    rows_batch = df.iloc[i:i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = get_embedding(rows_batch["Description"].tolist())
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'question_summary': row['Description'], 'question_raw': row['Patient'], 'answer_raw': row['Doctor']} for _, row in rows_batch.iterrows()]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))