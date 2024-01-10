import os
import openai
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
import time
import pinecone

load_dotenv()

def get_segments_from_bq():
    # Load the service account key JSON file.
    credentials = Credentials.from_service_account_file(
        "./asklex-f6e79ca5e230.json")

    # Use the credentials to build a BigQuery client.
    bigquery_client = bigquery.Client(
        credentials=credentials, project=credentials.project_id)

    sql = """
        SELECT * FROM asklex.lexfridman_pod_transcriptions
    """

    # Make an API request.
    try:
        query_results_df = bigquery_client.query(sql).to_dataframe()
    except Exception as e:
        print(e)

    return query_results_df

def create_embeddings(segments):
    batch_start_idx = 0
    metadata = []
    for i in range(len(segments)):
        meta = {
            "episode_id": segments["episode_id"].iloc[i],
            "title": segments["title"].iloc[i],
            "pub_date": segments["pub_date"].iloc[i],
            "segment_start": segments["segment_start"].iloc[i],
            "segment_end": segments["segment_end"].iloc[i],
            "text": segments["text"].iloc[i]
        }
        metadata.append(meta)

    ids = [str(i+1) for i in range(len(segments))]

    # Use openai module to create embeddings
    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    MODEL = "text-embedding-ada-002"

    batch_size = 1000
    for i in tqdm(range(0, len(segments), batch_size)):
        # Create embeddings
        try:
            embeddings_batch = openai.Embedding.create(
                input=segments["text"].iloc[i:i+batch_size].tolist(),
                engine=MODEL
            )
            # Create pinecone index
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV")
            )

            # check if 'openai' index already exists (only create index if not)
            if 'asklex' not in pinecone.list_indexes():
                pinecone.create_index('asklex', dimension=len(embeddings_batch["data"][0]["embedding"]))
            
            # connect to index
            index = pinecone.Index('asklex')

            # Comprehend the embeddings into an array
            embeds = [embedding["embedding"] for embedding in embeddings_batch["data"]]

            # Upsert the embeddings into the index in batches of 100
            pinecone_batch_size = 100
            pinecone_batch_start_idx = 0
            for i in range(0, len(embeds), pinecone_batch_size):
                current_batch_start_idx = pinecone_batch_start_idx + batch_start_idx
                # Get a batch of ids
                batch_ids = ids[current_batch_start_idx:current_batch_start_idx+pinecone_batch_size]

                # Get a batch of embeddings
                batch_embeds = embeds[pinecone_batch_start_idx:pinecone_batch_start_idx+pinecone_batch_size]

                # Get a batch of metadata
                batch_metadata = metadata[current_batch_start_idx:current_batch_start_idx+pinecone_batch_size]

                # Zip the ids, embeddings, and metadata
                to_upsert = zip(batch_ids, batch_embeds, batch_metadata)

                # Upsert the batch
                index.upsert(vectors=list(to_upsert))

                pinecone_batch_start_idx += pinecone_batch_size

            batch_start_idx += batch_size    
            time.sleep(2)
        except Exception as e:
            print(e)

def main():
    segments_df = get_segments_from_bq()
    create_embeddings(segments_df)

if __name__ == "__main__":
    main()
