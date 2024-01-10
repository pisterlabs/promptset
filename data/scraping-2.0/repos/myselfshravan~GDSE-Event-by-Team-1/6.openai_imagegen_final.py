import streamlit as st
import openai

st.set_page_config(page_title="OpenAI Image Gen", page_icon="🖼️")

openai.api_key = st.secrets["openai_api_key"]

st.title("AI Image Gen 🤖")
st.subheader("Write the Prompt")
prompt = st.text_area("Prompt", "A cute baby sea otter")

ask_button = st.button("Generate Image")
if ask_button:
    st.subheader("OpenAI Image Generator")
    completion = openai.Image.create(
        prompt=f"{prompt}",
        n=1,  # Number of images to create.
        size="512x512"  # Image size.
    )
    image_url = completion.data[0].url
    st.image(image_url)
