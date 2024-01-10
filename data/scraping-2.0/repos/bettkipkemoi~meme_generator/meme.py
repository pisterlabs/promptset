import streamlit as st
import requests
import openai
import random

st.set_page_config(page_title="Meme Generator", page_icon=":laughing:")

# Set your Unsplash API key here
UNSPLASH_API_KEY = "YOUR_API_KEY"

# Set your OpenAI GPT-3 API key here
GPT3_API_KEY = "YOUR_API_KEY"

# Function to generate a meme caption using GPT-3
def generate_meme_caption(prompt):
    openai.api_key = GPT3_API_KEY
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=30,
    )
    return response.choices[0].text

# Function to fetch a random image from Unsplash
def get_random_image(query):
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_API_KEY}",
    }
    params = {
        "query": query,
        "per_page": 1,
    }
    response = requests.get("https://api.unsplash.com/photos/random", headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Streamlit UI
st.title("Meme Generator")
st.sidebar.title("Generate Your Meme")

# Input for the meme topic
meme_topic = st.sidebar.text_input("Meme Topic", "funny cats")

# Button to generate a meme
if st.sidebar.button("Generate Meme"):
    # Fetch a random image
    image_data = get_random_image(meme_topic)
    if image_data:
        image_url = image_data["urls"]["regular"]
        st.image(image_url, caption="Your Random Image")

        # Generate a meme caption
        meme_caption = generate_meme_caption(f"Create a meme about {meme_topic}.")
        st.write("Meme Caption:", meme_caption)
    else:
        st.warning("Unable to fetch an image. Please try again with a different topic.")

# Add a footer
st.sidebar.markdown("Created by Your Name")

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Meme Generator!")
