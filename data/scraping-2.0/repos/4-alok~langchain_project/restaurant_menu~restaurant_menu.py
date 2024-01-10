import time
import streamlit as st
import os
import openai
from dotenv import load_dotenv, find_dotenv

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox(
    "Pick a Cuisine", ("Indian", "Punjab", "Bihar", "Odisha", "Tamil Nadu", "Assam",))

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_restaurant_name_and_item(cuisine):
    time.sleep(2)

    return {
        'restaurant_name' : 'Curry Delight',
        'menu_items' : 'samosa, paneer, tikka'
    }

if cuisine:
    res = generate_restaurant_name_and_item(cuisine)
    st.header(res['restaurant_name'])
    menu_items = res['menu_items'].split(',')
    st.write("Menu Items")
    for item in menu_items:
        st.write(item.strip())
