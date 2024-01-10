import os
import requests
from langchain.llms import CTransformers
import streamlit as st

MODEL_PATH = "llama-2-7b-chat.ggmlv3.q2_K.bin"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model..."):
        r = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

chat_model = CTransformers(model="llama-2-7b-chat.ggmlv3.q2_K.bin", model_type="llama")

st.header("Poet Struggling with Love")
st.write("Tell me about your feelings and emotions. I will express them in a poem for you.")

# Dropdown for type of relationship issue
issue_type = st.selectbox(
    "What type of relationship issue are you facing?",
    ["Choose an option", "Breakup", "Jealousy", "Communication Issues", "Distance", "Others"],
)

# Multiselect for feelings involved
feelings = st.multiselect(
    "What emotions are you feeling?",
    ["Sadness", "Anger", "Indifference", "Confusion", "Joy", "Others"],
)

# Text input for any additional information or specifics
content = st.text_input("Please share more details or your sentiments:")

# When the button is pressed or enter is hit after input
if content or st.button("Request a Poem"):
    # Generate a poem using all the information provided
    poem_request = f"Write a poem about the {issue_type} issue and the feelings of {', '.join(feelings)}. {content}"
    with st.spinner("Thinking of a poem..."):
        result = chat_model.predict(poem_request)
        st.write(result)
        print("poem_request: ", poem_request)
        print("result: ", result)
