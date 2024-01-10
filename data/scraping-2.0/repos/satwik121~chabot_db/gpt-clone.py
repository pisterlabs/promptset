import streamlit as st
import openai

# Set your OpenAI API key
api_key = "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"
openai.api_key = api_key

# Initialize chat history as an empty list
chat_history = []

# Define a function to interact with the chatbot
def chat_with_bot(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
    )
    return response.choices[0].text

# Streamlit app title and description
st.title("Chatbot with GPT-3")
st.write("Enter a message, and the chatbot will respond.")

# User input text box
user_input = st.text_input("You:", "")

# Bot response
if st.button("Send"):
    if user_input:
        prompt = f"You: {user_input}\nBot:"
        bot_response = chat_with_bot(prompt)

        # Add the user's input and bot's response to chat history
        chat_history.append((user_input, bot_response))

        st.write("Bot:", bot_response)
    else:
        st.write("Please enter a message.")

# Display chat history
st.subheader("Chat History")
for user_message, bot_message in chat_history:
    st.text(f"You: {user_message}")
    st.text(f"Bot: {bot_message}")

# Add a "Quit" button to exit the app
if st.button("Quit"):
    st.stop()
