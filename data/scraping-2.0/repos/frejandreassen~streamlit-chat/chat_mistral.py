import openai
import streamlit as st

## INSTRUCTIONS
# Open LMStudio and start Mistral server
# Start with streamlit run chat_mistral.py

# Set the title of the Streamlit app
st.title("Simple Mistral chat")

# Load OpenAI API key from Streamlit secrets
openai.base_url="http://localhost:1234/v1/"

# Define the GPT model to be used
# GPT_MODEL = "gpt-4"
GPT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Initialize session state for storing chat messages if not already set
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new message
user_input = st.chat_input("What's up?:")
if user_input:
    # Add user's message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the GPT-4 reply
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].finish_reason == "stop": 
                message_placeholder.markdown(full_response)
                break
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")

    # Add bot's reply to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
