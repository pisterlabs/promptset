import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

import os

import random

import uuid

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_csv_agent
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.llms import OpenAI

from sqlalchemy.sql import text
import pinecone
import numpy as np
import pickle
import pandas as pd


with open('pinecone_key.txt') as f:
    API_KEY = f.readlines()[0].strip()

pinecone.init(api_key=API_KEY, environment="us-west1-gcp-free")
# Initialize the pinecone index
index = pinecone.Index("agihouse")

positive_image_ids = []
negative_image_ids = []
NAMESPACE = 'test2'
# TODO: start with random centroid
fetch_response = index.fetch(ids=["1"], namespace=NAMESPACE)
id = list(fetch_response.vectors.keys())[0]
centroid = np.array(fetch_response.vectors[id]['values'])
negative_centroid = []
# print(centroid[0])





# TODO: Update after every turn
def get_gallery_images():
    return ["default_image.png"] * 13


class Product:
    def __init__(self, id, url, embedding):
        self.id = id
        self.url = url
        self.embedding = embedding

def update_gallery():
    print("Updating gallery")
    print(st.session_state['centroid'])
    query_response = index.query(
        namespace=NAMESPACE,
        top_k=13,
        include_values=True,
        include_metadata=True,
        vector=st.session_state['centroid'].tolist(),
    )
    st.session_state['gallery_images'] = []
    for i, match in enumerate(query_response['matches']):
        st.session_state['gallery_images'].append(match['metadata']['link'])
    print('Done updating')



def generate_frontend():
    main_col1, main_col2 = st.columns(2)

    # Use st.session_state to create a state for Streamlit that will contain a list of images
    if 'liked' not in st.session_state or 'not_liked' not in st.session_state:
        st.session_state['liked'] = []
        st.session_state['not_liked'] = []
    if 'centroid' not in st.session_state:
        st.session_state['centroid'] = np.array([0.0] * 1408)

    if 'gallery_images' not in st.session_state:
        st.session_state['gallery_images'] = get_gallery_images()

    if 'pickel' not in st.session_state:
        with open('embeddings.pkl', 'rb') as f:
            df = pickle.load(f)
            st.session_state['pickel'] = df

    if 'active_prod' not in st.session_state:
        st.session_state['active_prod'] = 0

    if 'seen' not in st.session_state:
        st.session_state['seen'] = set()


    # Left half - Tinder-like frontend
    with main_col1:
        st.subheader("Your Style")
        # images = get_gallery_images()
        used_list = st.session_state['gallery_images'][1:]
        for i in range(0, len(used_list), 3):
            row = st.columns(3)
            for j in range(3):
                with row[j]:
                    st.image(used_list[i + j], use_column_width=True)

    # Right half - Grid of images
    with main_col2:
        st.subheader("Swiper")
        next_prod_id = 1
        for i in range (240):
            if i not in st.session_state['seen']:
                next_prod_id = i
                break
        # print(next_prod_id)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("No", on_click=update_gallery):
                print("Previous button clicked")
                # TODO: Push the current image to the negative_image_ids list
                st.session_state['not_liked'].append(st.session_state['active_prod'])
                st.session_state['seen'].add(next_prod_id)
                for i in range (240):
                    if i not in st.session_state['seen']:
                        next_prod_id = i
                        break
                st.session_state['active_prod'] = next_prod_id

        with col2:
            if st.button("Yes", on_click=update_gallery):
                print("Next button clicked")
                curr = np.array(st.session_state['centroid'])
                new = st.session_state['pickel'].iloc[next_prod_id].embedding
                st.session_state['centroid'] = np.mean([curr, new], axis=0)
                st.session_state['seen'].add(st.session_state['active_prod'])
                st.session_state['liked'].append(st.session_state['active_prod'])
                for i in range (240):
                    if i not in st.session_state['seen']:
                        next_prod_id = i
                        break
                st.session_state['active_prod'] = next_prod_id
        st.session_state['active_prod'] = next_prod_id
        st.session_state['card_image'] = st.session_state['pickel'].iloc[next_prod_id].url
        card_image = st.image(st.session_state['card_image'], use_column_width=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display images in each column
    with col1:
        st.subheader("Liked Images")
        if len(st.session_state['liked']) > 0:
            cols = st.columns(len(st.session_state['liked']))
            for i, col in enumerate(cols):
                with col:
                    st.image(st.session_state['pickel'].iloc[st.session_state['liked'][i]].url, use_column_width=True)

    with col2:
        st.subheader("Disliked Images")
        if len(st.session_state['not_liked']) > 0:
            cols = st.columns(len(st.session_state['not_liked']))
            for i, col in enumerate(cols):
                with col:
                    st.image(st.session_state['pickel'].iloc[st.session_state['not_liked'][i]].url, use_column_width=True)


generate_frontend()
