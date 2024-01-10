from openai import OpenAI
import streamlit as st

# Set the title of the Streamlit app
st.title("Simple chat")

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Define the GPT model to be used
GPT_MODEL = "gpt-4-1106-preview"

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
        completion = client.chat.completions.create(
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
