'''
Author: Georgio Feghali
Date: July 11 2023
'''

# UI Dependencies
import streamlit as st
from PIL import Image

# Logic Dependencies
import openai

# Page Configuration.
favicon = Image.open("./admin/branding/logos/favicon-32x32.png")
st.set_page_config(
    page_title="Solvr.ai - Image Generation",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

## Logic
def create_image(prompt):
    openai.api_key = st.secrets['openai_api_key']
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    img_url = response['data'][0]['url']

    return img_url

# UI
st.markdown("<h1 style='text-align: center; vertical-align: middle;'>Image Generator 🖼️</h1>", unsafe_allow_html=True)

prompt = st.text_input("Prompt", placeholder="Describe the image you want us to generate for you!")
if st.button("Generate!"):
    if prompt == "":
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating Image..."):
            img_url = create_image(prompt)
            st.text(img_url)
