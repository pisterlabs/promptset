import streamlit as st
from openai import OpenAI
import consts
import os
import tempfile

def show_audio(message): 
    speech_file_path = tempfile.NamedTemporaryFile(delete=True).name
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=message
    )
    response.stream_to_file(speech_file_path)
    st.audio(speech_file_path)

# Placeholder for chat messages
chat_container = st.empty()
os.environ['OPENAI_API_KEY'] = consts.API_KEY_OPEN_AI
client = OpenAI()

def show_chat_history():
    if len(st.session_state.chat_history) == 0:
        st.write("")
        return

    for entry in st.session_state.chat_history:
        if len(entry) == 2:  # Check if the entry has exactly two elements
            author, message = entry
            with st.chat_message(author):
                st.write(message)
        else:
            st.error(f"Invalid entry in chat history: {entry}")

def get_ai_response(prompt):
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt + ". Make it quite short briefly (maximum 200 words), smart, relevant, straightforward and use table to describe",
            temperature=0.4,
            max_tokens=400
        )
        return response.choices[0].text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("AI Chatbot")
# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
# Chat input for user message
user_message = st.chat_input("Type your message...")

if user_message:
    # Add user message to chat history
    st.session_state.chat_history.append(('user', user_message))

    # Temporary loading message
    loading_message = "AI is writing..."
    st.session_state.chat_history.append(('AI Assistant', loading_message))

    # Display chat history including the loading message
    for author, message in st.session_state.chat_history:
        with st.chat_message(author):
            st.write(message)

    # Get AI response
    ai_response = get_ai_response(user_message)

    # Replace the loading message with the actual response
    st.session_state.chat_history[-1] = ('AI Assistant', ai_response)

    # Redisplay the chat history with the actual response
    st.experimental_rerun()

show_chat_history()  # Show the chat history

# Clear chat history button
if len(st.session_state.chat_history) > 0:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        show_chat_history()  # Show the empty chat history
        chat_container.empty()  # Clear the chat input box
        st.experimental_rerun()

