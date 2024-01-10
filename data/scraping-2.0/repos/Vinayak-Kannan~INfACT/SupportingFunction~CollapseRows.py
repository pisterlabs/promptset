import uuid

import numpy as np
import openai
from dotenv import dotenv_values
from openai.embeddings_utils import cosine_similarity, get_embedding
from pandas import DataFrame
import pandas as pd
import pinecone


def get_embeddings(texts):
    embeddings = []
    for text in texts:
        embedding_model = "text-embedding-ada-002"
        value = get_embedding(text, engine=embedding_model)
        embeddings.append(value)
    return embeddings


# Ideas to do this better:
# 1. Store the embeddings in vectorDB to stop having to call the API every time
# 2. Instead of dropping terms, roll them up into a list of terms, then have LLM synthesize them all into a single term
# 3. Switch to LLAMA for better results, fine tune it on accruate responses we have
def collapse_rows(df: DataFrame, school) -> DataFrame:
    config = dotenv_values("/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/.env")
    openai.api_key = config.get("SECRET_KEY")
    # Get embeddings for each value in the 'Skill' column
    embeddings = get_embeddings(df['Skill'].tolist())
    df['Embedding'] = embeddings

    # Add a column to the df called 'Collapsed Skill'
    df['Collapsed Skill'] = ''

    # Iterate through the similarity matrix
    for i, row in enumerate(df['Embedding']):
        if df.loc[i, 'Collapsed Skill'] != "":
            continue
        similarities = []
        for j, row2 in enumerate(df['Embedding']):
            # Get skill name from the 'Skill' column
            skill_name = df.loc[j, 'Skill']
            if row is not None and row2 is not None:
                # Calculate the cosine similarity between the two rows
                similarities.append([cosine_similarity(row, row2), skill_name, j])

        # Filter similarities to only include values that are greater than 0.8
        similarities = [x for x in similarities if x[0] >= 0.85]

        count = 0
        if len(similarities) > 0:
            word = similarities[0][1]
            for similarity in similarities:
                if similarity[1] == word:
                    count += 1

        if count > 0 and count == len(similarities):
            df.loc[i, 'Collapsed Skill'] = df.loc[i, 'Skill']
            continue

        # Create a string that concats similarities
        similar_skills = df.loc[i, 'Skill'] + ", "
        for similarity in similarities:
            similar_skills += similarity[1] + ", "
        # Create prompt that asks OpenAI for a label that summarizes the similar skills
        course_description = f"""
            Review the following terms, seperated by commas, and summarize them with one label. 
            
            Follow the format as per example below:
            The terms are: Bowling, Skiing and Surfing, Vibing
            
            Label: Sporting Activities
            
            The terms are: {similar_skills}
        """

        prompt_message = [{
            "role": "user",
            "content": course_description
        }]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt_message,
            temperature=0
        )

        response_message = response.choices[0].message.content
        print(similar_skills)
        # Get the label from the response
        label = response_message.split("Label: ")[1].strip()

        # Check if similar_skills are all the same when lowered to lower case and removing punctuation
        # If they are, then set the label to the first skill
        # If they are not, then set the label to the response
        similar_skills = similar_skills.lower().replace(",", "").replace("-", "").replace(" ", "").split()
        label = similar_skills[0] if len(set(similar_skills)) == 1 else label
        print(label)

        # If there are similar values, add the first similar value to the 'Collapsed Skill' column
        df.loc[i, 'Collapsed Skill'] = label
        for similarity in similarities:
            if df.loc[similarity[2], 'Related Course'] != df.loc[i, 'Related Course']:
                df.loc[similarity[2], 'Collapsed Skill'] = label
            else:
                df.loc[similarity[2], 'Collapsed Skill'] = None

    # Drop the 'Embedding' column
    df = df.drop(columns=['Embedding'])

    # Merge original column with df
    orig_df = pd.read_csv(
        f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/{school}/Data - Sheet1.csv')
    # Drop credits and syllabus from orig_df
    orig_df = orig_df.drop(columns=['Credits', 'Syllabus'])
    df = pd.merge(df, orig_df, left_on=['Related Course', 'Semester'], right_on=['Title', 'Semester'])

    return df


def collapse_rows_pinecone(df: DataFrame):
    config = dotenv_values("/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/.env")
    openai.api_key = config.get("SECRET_KEY")

    # Convert rows in 'Credits' column to numeric
    df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce')
    # Filter df to rows where credits value contains a number even if it is a string
    df = df[df['Credits'].notna()]

    df['Embedding'] = ''
    print(len(df))
    for i, row in df.iterrows():
        embedding_model = "text-embedding-ada-002"
        print(f"Attribute: {row['Skill']}. Explanation of how course teaches this attribute: {row['Explanation']}")
        embedding = get_embedding(row['Skill'] + row['Explanation'], engine=embedding_model)
        df['Embedding'].update(pd.Series([embedding], index=[i]))

    config = dotenv_values("/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/.env")
    openai.api_key = config.get("SECRET_KEY")
    # get api key from app.pinecone.io
    PINECONE_API_KEY = config.get('PINECONE_KEY')
    # find your environment next to the api key in pinecone console
    PINECONE_ENV = config.get('PINECONE_ENV')

    # Send to pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = pinecone.GRPCIndex("infact")
    df['Collapsed Skill'] = ''
    df['Reviewed'] = False

    for i, row in df.iterrows():
        if row['Reviewed']:
            continue
        # Find all rows in index that have a cosine similarity greater than 0.85
        embedding = row['Embedding']
        similar_rows = [[row['Skill'], row['Embedding'], i]]
        for j, row_compare in df.iterrows():
            if cosine_similarity(embedding, row_compare['Embedding']) > 0.85:
                similar_rows.append([row_compare['Skill'], row_compare['Embedding'], j])
                break

        # Get center of all embeddings
        similar_row_embeddings = []
        for row in similar_rows:
            df.at[row[2], 'Reviewed'] = True
            similar_row_embeddings.append(row[1])
        embeddings = np.array(similar_row_embeddings)  # replace with your array of embeddings
        centroid = np.mean(embeddings, axis=0)

        # Find the closest row in similar_rows to the center of all embeddings
        closest_row = similar_rows[0]
        for row_compare in similar_rows:
            if cosine_similarity(centroid, row_compare[1]) > cosine_similarity(centroid, closest_row[1]):
                closest_row = row_compare

        upsert_value = [{
            "id": uuid.uuid4().hex,
            "values": closest_row[1],
            "metadata": {'text': closest_row[0]}
        }]
        response = index.query(vector=closest_row[1], top_k=1, include_values=False, include_metadata=True).to_dict()
        if len(response['matches']) == 0 or response['matches'][0]['score'] < 0.85:
            index.upsert(vectors=upsert_value, show_progress=True)

    for i, row in df.iterrows():
        embedding = row['Embedding']
        response = index.query(vector=embedding, top_k=1, include_values=False, include_metadata=True).to_dict()
        if len(response['matches']) == 0:
            upsert_value = [{
                "id": uuid.uuid4().hex,
                "values": embedding,
                "metadata": {'text': row['Skill']}
            }]
            index.upsert(vectors=upsert_value, show_progress=True)
            df.loc[i, 'Collapsed Skill'] = row['Skill']
            continue
        top_match = response['matches'][0]
        if top_match['score'] > 0.85:
            df.loc[i, 'Collapsed Skill'] = top_match['metadata']['text']
        else:
            print(str(top_match['score']) + " :" + str(top_match['metadata']['text']))
            print(str(row['Skill']))
            print("RIP")
            df.loc[i, 'Collapsed Skill'] = row['Skill']
            upsert_value = [{
                "id": uuid.uuid4().hex,
                "values": embedding,
                "metadata": {'text': row['Skill']}
            }]
            index.upsert(vectors=upsert_value, show_progress=True)

    # Drop df embedding column
    df = df.drop(columns=['Embedding'])
    df = df.drop(columns=['Reviewed'])
    return df
