import streamlit as st
import openai
from decouple import config

# Read API key and API base from .env file
api_key = config('API_KEY')
api_base = config('API_BASE')

# Set OpenAI configuration
openai.api_type = "azure"
openai.api_base = api_base
openai.api_version = "2022-12-01"
openai.api_key = api_key

st.title("OpenAI Chatbot App (Streamlit)")

# Create and configure a chat display area
#chat_display = st.text_area("Chat Display", value="", height=400, max_chars=None, key="chat_display", disabled=True)

# Create an input field for user messages
user_message = st.text_input("your prompt")

# Create a button to send the message
if st.button("Send"):
    # Display the user's message
    st.markdown("You: " + user_message)

    # Generate a response from the chatbot
    response = openai.Completion.create(
        engine="gpt35turbo",
        prompt=user_message,
        temperature=0.2,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.5,
        stop=None
    )

    bot_message = response.choices[0].text

    # Display the chatbot's response
    st.markdown("Bot: " + bot_message)

# Note: In Streamlit, there's no need to start a main loop; it's handled automatically.

