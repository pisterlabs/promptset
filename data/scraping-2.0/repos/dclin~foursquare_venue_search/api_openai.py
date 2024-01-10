import openai 
import streamlit as st 

openai.api_key = st.secrets['openai']['api_key'] 


def get_embedding(category_str):
    try:
        response = openai.Embedding.create(
            input=category_str,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        return embeddings 
    
    except Exception as e: 
        raise e 



def get_moderation(user_query):
    try:
        moderation = openai.Moderation.create(
            input=user_query
        )
        moderation_result = moderation['results'][0]
        flagged_categories = [category for category, value in moderation_result['categories'].items() if value]
        return {'flagged': moderation_result['flagged'], 'flagged_categories':flagged_categories}
    
    except Exception as e: 
        raise e 
    
