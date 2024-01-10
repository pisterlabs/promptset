import streamlit as st
import pandas as pd
import os
import numpy as np
import openai


openai.api_key = os.getenv('OPENAI_API_KEY')

st.title("Feature Extraction")

path_embeddings = "embeddings"
path_journals = "journals"

prompt = "Read"

# define call function for openai
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = temperature,
    )
    return response.choices[0].message["content"]

# load embeddings into cache
@st.cache_data
def load_embeddings(filename):
    data = pd.read_csv(f'{path_embeddings}/{filename}')
    return data

# load journal into cache
@st.cache_data
def load_journal(filename):
    data = pd.read_csv(f'{path_journals}/{filename}')
    return data


def extract_features(name, values):

    return



if __name__ == "__main__":
     
    st.text("! under construction -> jupyter files !")
    filename_embeddings = st.sidebar.selectbox(label= 'List of Embeddings', options=os.listdir(path_embeddings))
    filename_journals = st.sidebar.selectbox(label= 'List of Journals', options=os.listdir(path_journals))
    if st.sidebar.button(label="Load files", type="primary"):
        data_embeddings =  load_embeddings(filename_embeddings)
        data_journals = load_journal(filename_journals)
        st.sidebar.success("Your files have been loaded. You can search them now.")

        st.write(data_journals)
        
        #data_journals['Sentiment'] = data_journals['Text'].apply(lambda x: get_completion(x, temperature=0.5))

        # go over journal entries. Send each entry to openai and get json back.
        # 
        for name, values in data_journals['Text'].items():
            st.write(values)

