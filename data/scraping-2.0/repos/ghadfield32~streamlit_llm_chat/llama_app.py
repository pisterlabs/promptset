import openai
import streamlit as st

# Page configuration.
st.set_page_config(
    page_title="🏀 Chat with a Basketball Pro!",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Set the OpenAI API key.
openai.api_key = st.secrets["openai_key"]

# Display title and basketball-themed info.
st.title("🏀 Chat with an Old Basketball Pro!")
st.write("Hey there! Back in my day, I used to sink threes with the best of 'em. Now, I'm here to help you with anything basketball. What's on your mind, rookie?")

# Initialize session state for messages if not already done.
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "Warming up on the court! 🏀 Ready to assist and share some hoops wisdom. Pass the ball, and let's get this conversation rolling!"}]

# Display previous messages.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# When the user submits a new question or statement.
if prompt := st.chat_input():
    try:
        # Append user's message to the messages list
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Request a completion from the model
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        
        response_text = response.choices[0].message["content"].strip()
        
        # Append the model's response to the messages list
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
    except openai.error.RateLimitError:
        st.warning("Sorry, we've dribbled too much and need a timeout. Please try again later!")
