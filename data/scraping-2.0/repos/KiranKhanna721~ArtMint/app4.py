import openai
import streamlit as st    
import requests
from io import BytesIO



def generate_nft_artwork(description):
    prompt = f"Create an NFT artwork based on the description: {description}"
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']
    return image_url

def app():
    st.header("NFT Art Generation with DALL·E")
    openai.api_key =  st.secrets["OPENAI_API_KEY"]
    description = st.text_input("Description for a image you want to generate")
    image_url = generate_nft_artwork(description)
    st.image(image_url)
    if image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Downloaded Image")
        else:
            st.write("Error: Invalid image URL")
