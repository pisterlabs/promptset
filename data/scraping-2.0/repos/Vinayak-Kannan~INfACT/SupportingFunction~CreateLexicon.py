import json

import numpy as np
import openai
from dotenv import dotenv_values
from openai.embeddings_utils import cosine_similarity, get_embedding
import uuid
import pandas as pd
import pinecone


def create_lexicon(generate: bool = False):
    config = dotenv_values("/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/.env")
    openai.api_key = config.get("SECRET_KEY")
    # get api key from app.pinecone.io
    PINECONE_API_KEY = config.get('PINECONE_KEY')
    # find your environment next to the api key in pinecone console
    PINECONE_ENV = config.get('PINECONE_ENV')

    if generate:
        # Get a list of all skills, knowledge, abilities
        skills_columbia_fall = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/SkillOutputv2.csv')['Skill'].tolist()
        skills_columbia_spring = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Spring2023_v1/SkillOutputv2.csv')['Skill'].tolist()

        knowledge_columbia_fall = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/KnowledgeOutputv2.csv')['Skill'].tolist()
        knowledge_columbia_spring = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Spring2023_v1/KnowledgeOutputv2.csv')['Skill'].tolist()

        abilities_columbia_fall = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/AbilitiesOutputv2.csv')['Skill'].tolist()
        abilities_columbia_spring = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Spring2023_v1/AbilitiesOutputv2.csv')['Skill'].tolist()

        # Combine all lists into a dataframe with one column
        key_words = skills_columbia_fall + skills_columbia_spring + knowledge_columbia_fall + knowledge_columbia_spring + abilities_columbia_fall + abilities_columbia_spring
        key_words = list(set(key_words))

        # Create a new dataframe with three columns: 'id' and 'values' and 'metadata'
        df = pd.DataFrame(columns=['id', 'values', 'metadata'])
        for i, word in enumerate(key_words):
            embedding_model = "text-embedding-ada-002"
            embedding = get_embedding(word, engine=embedding_model)
            df.loc[i] = [uuid.uuid4().hex, embedding, {'text': word}]
        # Save to csv
        df.to_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/lexicon.csv', index=False)
    else:
        # Read from csv
        df = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/lexicon.csv')

    # Go through each row in dataframe and convert metadata to a dictionary
    # for i, row in df.iterrows():
    #     # Replace ' with " in metadata
    #     row['metadata'] = row['metadata'].replace("'", '"')
    #     df.loc[i, 'metadata'] = json.loads(row['metadata'])

    # Loop through each row and see if there is a similar row with greater than 0.85 cosine similarity. If so, drop current row
    for i, row in df.iterrows():
        for j, row2 in df.iterrows():
            if row['values'] == row2['values']:
                print("same")
                continue
            if cosine_similarity(row['values'], row2['values']) > 0.85:
                df.drop(i, inplace=True)
                break


    # Send to pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = pinecone.GRPCIndex("infact")
    dict_to_upsert = df.to_dict('records')
    # Split up dict_to_upsert into 100 record chunks
    chunks = [dict_to_upsert[x:x + 100] for x in range(0, len(dict_to_upsert), 100)]
    for chunk in chunks:
        index.upsert(vectors=chunk, show_progress=True)