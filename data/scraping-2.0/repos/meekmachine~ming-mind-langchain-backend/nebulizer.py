import os
import pandas as pd
from convokit import Corpus, download
import openai
from firebase_admin import credentials, firestore, initialize_app
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import redis
import pickle
import json
from dotenv import load_dotenv
import firebase_admin

load_dotenv()

cred = credentials.Certificate(".keys/ming-527ed-firebase-adminsdk-z38ui-27b7e06411.json")
firebase_admin.initialize_app(cred)


db = firestore.client()

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")



# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)

def calculate_embeddings(text):
    response = openai.Embedding.create(input=text, engine="text-similarity-babbage-001")
    return response['data'][0]['embedding']

def load_corpus():
    # Check if DataFrames are in Redis
    conversations_df = r.get('conversations_df')
    utterances_df = r.get('utterances_df')
    speakers_df = r.get('speakers_df')

    if conversations_df is None or utterances_df is None or speakers_df is None:
        try:
            # Download and load the corpus
            corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
            conversations_df = corpus.get_conversations_dataframe()
            utterances_df = corpus.get_utterances_dataframe()
            speakers_df = corpus.get_speakers_dataframe()

            # Serialize and store the DataFrames in Redis
            r.set('conversations_df', pickle.dumps(conversations_df))
            r.set('utterances_df', pickle.dumps(utterances_df))
            r.set('speakers_df', pickle.dumps(speakers_df))
        except json.JSONDecodeError:
            print("Error loading corpus. Please check the corpus data.")
            return None, None, None
    else:
        # Deserialize the DataFrames
        conversations_df = pickle.loads(conversations_df)
        utterances_df = pickle.loads(utterances_df)
        speakers_df = pickle.loads(speakers_df)

    return conversations_df, utterances_df, speakers_df

def process_conversations(conversations_df, utterances_df, speakers_df):
    # Filter and merge dataframes
    merged_df = pd.merge(utterances_df, speakers_df, left_on='speaker', right_index=True)
    merged_df = pd.merge(merged_df, conversations_df, left_on='conversation_id', right_index=True)

    # Process each conversation
    for convo_id, convo_data in merged_df.groupby('conversation_id'):
        # Calculate total personal attacks and average toxicity
        total_attacks = convo_data['meta.comment_has_personal_attack'].sum()
        avg_toxicity = convo_data['meta.toxicity'].mean()

        # Get conversation start time
        start_time = convo_data['timestamp'].min()

        # Calculate embeddings for the conversation
        convo_text = ' '.join(convo_data['text'].tolist())
        embeddings = calculate_embeddings(convo_text)

        # Prepare data for Firebase
        convo_record = {
            'convo_id': convo_id,
            'total_personal_attacks': total_attacks,
            'average_toxicity': avg_toxicity,
            'start_time': start_time,
            'embeddings': embeddings,
            # Include additional metadata as needed
        }

        # Save to Firebase
        db.collection('conversations').document(convo_id).set(convo_record)

    print("Conversations processed and saved to Firebase.")


conversations_df, utterances_df, speakers_df = load_corpus()

if conversations_df is not None:
    process_conversations(conversations_df.head(100), utterances_df, speakers_df)
else:
    print("Failed to load corpus data.")
