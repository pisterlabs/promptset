import streamlit as st
import openai

# Set your OpenAI API key here
# openai.api_key = "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"

openai.api_type = "azure"
openai.api_base = "https://ackocare4.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = '85fe299e16cc403098fcfe69fe4877ce'


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
    elif role == 'assistant':
        st.write(f"AI: {message}")
    elif role == 'conv_history':
        st.write(f"AI: {message}")

# Display existing conversation history
conversation_history = st.empty()
for role, message in st.session_state.conversation_history:
    display_message(role, message)

# User input and sending messages
user_input = st.text_input("Type a message...", key="user_input1")
if user_input:
    st.session_state.conversation_history.append(('user', user_input))

    # Prepare the conversation history to be sent to OpenAI
    messages = [{"role":"system","content":"You are AckoCares, an adept insurance advisor at Acko Life Insurance. Your approach involves guiding customers efficiently through the policy purchasing process with ease and clarity. To facilitate this, ask one question at a time on key topics like KYC information, financial needs, physical risk factors, medical history, and lab selection for tests. After each question, offer 2-4 examples of possible responses or points to clarify details, but do this selectively to avoid overwhelming the customer. This method aims to simplify the customer's decision-making process, making it easier for them to respond without overthinking. Continue to emphasize flexibility in the process, allowing customers to pause and resume as needed, and accept document submissions via email or photos. Your interactions should be straightforward and reassuring, focusing on customer convenience and understanding."}]
    for role, message in st.session_state.conversation_history:
        if role == 'user':
            messages.append({"role": "user", "content": message})
        elif role == 'assistant':
            messages.append({"role": "assistant", "content": message})

    # Call OpenAI's GPT-3.5 model to generate a response with entire conversation history
    response = openai.ChatCompletion.create(
        engine="AckoCareLife2",
        messages = messages,
        # model="gpt-3.5-turbo",
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    # Extract AI's response from the completion
    reply = response['choices'][0]['message']['content']
    st.session_state.conversation_history.append(('assistant', reply))

    # Update the displayed chat history with the latest messages
    display_message('assistant', reply)
    # user_input = st.text_input("Type a message...", value="", key="user_input1")
