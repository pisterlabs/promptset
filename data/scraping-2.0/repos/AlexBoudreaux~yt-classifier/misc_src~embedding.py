import pandas as pd
import numpy as np
from openai import OpenAI
from supabase import create_client


SUPABASE_URL = 'https://bbrcyfqrvwqbboudayre.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJicmN5ZnFydndxYmJvdWRheXJlIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTQ1Nzc0MzAsImV4cCI6MjAxMDE1MzQzMH0.SPNLpnm_cIHUdYMOKOK4d56VmgfNpuTComWRigMBwTg'

client = OpenAI(api_key="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")

def get_videos_from_supabase(url, key):
    supabase = create_client(url, key)
    cooking_videos = []
    start_idx = 0

    while True:
        table = supabase.table('videos')
        full_response = table.select('*').ilike('category', '%Cooking%').range(start_idx, start_idx + 1000 - 1).execute()

        response_data = full_response.data
        if not response_data:
            break

        cooking_videos.extend(response_data)
        start_idx += 1000

    return pd.DataFrame(cooking_videos)

videos_df = get_videos_from_supabase(SUPABASE_URL, SUPABASE_KEY)


from openai import OpenAI
import openai

def truncate_text(text, max_tokens=8090):
    # Tokenize and truncate if necessary
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text

def get_embedding(text, model="text-embedding-ada-002"):
    text = truncate_text(text.replace("\n", " "))  # Apply truncation after replacing newlines
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        print(f"Embedding computed: {embedding[:10]}...")  # Print first few elements of embedding for verification
        return embedding
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def compute_embeddings(videos_df, num_videos=15):
    for index, row in videos_df.iterrows():
        # Check if embedding already exists for this video
        # If 'embedding' is an array, use `all()` to check if all elements are NaN
        if isinstance(row['embedding'], list):
            if all(pd.isna(v) for v in row['embedding']):
                combined_text = f"{row['video_name']} by {row['creator']} {row['description']} {row['summary']} {row['transcript']}"
                embedding = get_embedding(combined_text)
                if embedding is not None:
                    videos_df.at[index, 'embedding'] = embedding
                else:
                    print(f"Failed to compute embedding for video ID {row['id']}")
            else:
                print(f"Embedding already exists for video ID {row['id']}")
        else:
            # If 'embedding' is not an array, use `pd.isna()` directly
            if pd.isna(row['embedding']):
                combined_text = f"{row['video_name']} by {row['creator']} {row['description']} {row['summary']} {row['transcript']}"
                embedding = get_embedding(combined_text)
                if embedding is not None:
                    videos_df.at[index, 'embedding'] = embedding
                else:
                    print(f"Failed to compute embedding for video ID {row['id']}")
            else:
                print(f"Embedding already exists for video ID {row['id']}")

    return videos_df

# Rest of your code for processing embeddings
processed_videos_df = compute_embeddings(videos_df)

# Function to update a video record in the database with its embedding
def update_video_with_embedding(supabase_client, video_id, embedding):
    try:
        # Check if embedding is already a list or needs conversion
        embedding_to_store = embedding if isinstance(embedding, list) else embedding.tolist() if embedding is not None else None
        # Update the record in the database
        updated = supabase_client.table('videos').update({'embeddings': embedding_to_store}).eq('id', video_id).execute()
        if updated.error:
            print(f"Error updating video ID {video_id}: {updated.error.message}")
        else:
            print(f"Successfully updated video ID {video_id}")
    except Exception as e:
        print(f"Exception while updating video ID {video_id}: {e}")



# Initialize Supabase client
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Iterate over the DataFrame and update each video record with its embedding
for index, row in processed_videos_df.iterrows():
    update_video_with_embedding(supabase_client, row['id'], row['embeddings'])

