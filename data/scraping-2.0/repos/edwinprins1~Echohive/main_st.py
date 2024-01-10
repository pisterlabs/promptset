import openai
import sys
import tiktoken
import streamlit as st

# Define the function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Set the maximum token limit
MAX_MEMORY_TOKENS = 100

# Set up session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

def chatbot_response(chat_input, placeholder_response):
    # Append user input to the conversation history
    st.session_state.conversation_history.append({"role": "user", "content": chat_input})
    
    # Calculate the total tokens in the conversation history
    total_tokens = sum(count_tokens(message["content"]) for message in st.session_state.conversation_history)

    # Remove the oldest message from conversation history if total tokens exceed the maximum limit
    while total_tokens > MAX_MEMORY_TOKENS:
        if len(st.session_state.conversation_history) > 2:
            removed_message = st.session_state.conversation_history.pop(1)
            total_tokens -= count_tokens(removed_message["content"])
        else:
            break

    # Make API calls to OpenAI with the conversation history and use streaming responses
    response = openai.ChatCompletion.create(
        model="gpt-4", # or gpt-3.5-turbo
        messages=st.session_state.conversation_history,
        stream=True,
    )

    # Process the response from the API
    assistant_response = ""
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue
        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            assistant_response += r_text
            placeholder_response.markdown(assistant_response, unsafe_allow_html=True)

    # Append the assistant's response to the conversation history
    st.session_state.conversation_history.append({"role": "assistant", "content": assistant_response})

# Streamlit app
st.title("Chatbot")
user_input = st.text_input("You:")
response_button = st.button("Send")
placeholder_response = st.empty()

if response_button:
    chatbot_response(user_input, placeholder_response)
