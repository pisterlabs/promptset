import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai
import os
import pinecone
import requests
from bs4 import BeautifulSoup as bs

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='asia-southeast1-gcp-free')
openai.api_key = os.getenv("OPENAI_API_KEY")
index = pinecone.Index("clubs-index")

def search_engine(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = response['data'][0]['embedding']
    query_response = index.query(
        namespace='uconn-clubs',
        top_k=5,
        include_values=True,
        include_metadata=True,
        vector=query_embedding,
    )
    
    ret = []
    for m in query_response["matches"]:
        temp = dict()
        temp["url"] = m["id"]
        temp["name"] = m["metadata"]["Club Name"]
        temp["short_desc"] = m["metadata"]["Club Short Description"]
        temp["long_desc"] = m["metadata"]["Club Long Description"]
        # temp["image_url"] = driver.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
        ret.append(temp.copy())
    
    return ret

def display_card(title, short, description, url):
    card_style = """
        <style>
            .card {
                border: 1px solid #d4d4d4;
                border-radius: 4px;
                padding: 10px;
                box-shadow: 2px 2px 12px #aaa;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }            
        </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)
    
    card_html = f"""
         <div class="card">
          <div class="content">
            <a href="{url}" alt="club link"><h2>{title}</h2></a>
             <h3>{short}</h3>
            <p>{description}</p>
          </div>
        </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
        


st.set_page_config(
    page_title="UCONN Clubs Search Engine", page_icon="🐍", layout="wide"
)
st.title("UCONN Clubs Search Engine")

query = st.text_input("Enter your search query:")

if query:
    results = search_engine(query)
    
    if results:
        st.write("Search Results:")
        for r in results:
            display_card(r["name"], r["short_desc"], r["long_desc"], r["url"])
    else:
        st.write("No results found!")
