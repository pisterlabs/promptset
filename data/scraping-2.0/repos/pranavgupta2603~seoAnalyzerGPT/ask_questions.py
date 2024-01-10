import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import streamlit as st
openai.api_key = st.secrets["openai_api_key"]


df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def create_context(
    question, df, max_len=1800, size="ada"
):
    
   # Create a context for a question by finding the most similar context from the dataframe
    

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=3000,
    size="ada",
    debug=False,
    max_tokens=500,
    stop_sequence=None
):
    #Answer a question based on the most similar context from the dataframe texts
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"You are an SEO anaylzer. \nYou will be given:\n An issue, \nthe description of the issue\nthe items that cause the issue.\n Describe the Issue. Show the appropriate solution to the issue. Implement the solution to the issue and show the fix on the given items in the issue. Give only the fix to the item given in the issue. \nContext: {context}\n\n---\n\nIssue: {question}",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

