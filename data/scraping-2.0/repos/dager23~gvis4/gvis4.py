import streamlit as st
from openai import OpenAI
import base64
import requests

client = OpenAI(api_key=st.secrets["APIKEY"],organization=st.secrets["ORG"])

def response(image,text):
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{text}"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image}",
                "detail": "low"
            },
            },
        ],
        }
    ],
    max_tokens=85,
    )

    return response.choices[0].message.content

def encode_image(image):
    return base64.b64encode(image).decode('utf-8')

# Path to your image
image=st.file_uploader("Upload an image here", type=['jpg','jpeg','png'])
if image:
    base64_image = encode_image(image.getvalue())
    if "messages" not in st.session_state:
                st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if text:= st.chat_input("Ask query regarding the image"):
        # Display user message in chat message container
        st.chat_message("user").markdown(text)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": text})

        response = f"Bot: {response(base64_image,text)}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})



