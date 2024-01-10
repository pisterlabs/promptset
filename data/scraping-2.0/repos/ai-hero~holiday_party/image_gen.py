import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Streamlit interface
st.title("Image Generator using DALL-E")

# User input for the prompt
user_prompt = st.text_input("Enter your prompt:", "a white siamese cat")

# Button to generate image
if st.button("Generate Image"):
    # API call to OpenAI
    response = client.images.generate(
        model="dall-e-3",
        prompt=user_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # Get the image URL from the response
    image_url = response.data[0].url

    # Fetch and display the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, caption="Generated Image")
