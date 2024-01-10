import streamlit as st
import openai

# Set your OpenAI API key here
openai.api_key = "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"

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Display title and chat history
st.title("Acko Care")
st.subheader("Welcome to your personal Life Insurance Helper! Answer the following questions to secure yourself")

# Define function to display messages
def display_message(role, message):
    if role == 'user':
        st.write(f"You: {message}")
    elif role == 'AI':
        st.write(f"AI: {message}")

# Display existing conversation history
conversation_history = st.empty()
for role, message in st.session_state.conversation_history:
    display_message(role, message)

# User input and sending messages
user_input = st.text_input("Type a message...", key="user_input")
if st.button("Send"):
    if user_input:
        st.session_state.conversation_history.append(('user', user_input))

        # Call OpenAI's GPT-3.5 model to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are chatting with an AI."},
                {"role": "user", "content": user_input}
            ],
        )

        # Append AI's response to the conversation history
        reply = response['choices'][0]['message']['content']
        st.session_state.conversation_history.append(('AI', reply))

        # Update the displayed chat history with the latest messages
        display_message('AI', reply)
