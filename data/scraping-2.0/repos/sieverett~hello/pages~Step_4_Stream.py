import os
import streamlit as st
import openai

# Display the chat history
def create_chat_area(chat_history):
    for chat in chat_history:
        role = chat['role']
        with st.chat_message(role):
            st.write(chat['content'])

# Generate chat responses using the OpenAI API
def chat(messages, max_tokens, temperature=1, n=1, model="gpt-3.5-turbo-16k", stream=False):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stream=stream
    )
    for chunk in completion:
        try:
            yield chunk.choices[0].delta.content if chunk.choices[0].finish_reason != "stop" else ''
        except:
            yield 'error!'

# Main function to run the Streamlit app
def main():
    # Streamlit settings
    st.markdown("""<style>.block-container{max-width: 66rem !important;}</style>""", unsafe_allow_html=True)
    st.title("ChatGpt Streamlit Demo")
    st.markdown('---')

    # Session state initialization
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False

    # API key setup
    openai_key = st.secrets["OPENAI_API_KEY"]
    if openai_key is None:
        with st.sidebar:
            st.subheader("Settings")
            openai_key = st.text_input("Enter your OpenAI key:", type="password")
    elif openai_key:
        run_chat_interface()
    else:
        st.error("Please enter your OpenAI key in the sidebar to start.")

# Run the chat interface within Streamlit
def run_chat_interface():
    create_chat_area(st.session_state.chat_history)

    # Chat controls
    clear_button = st.button("Clear Chat History") if len(st.session_state.chat_history) > 0 else None
    user_input = st.chat_input("Ask something:")

    # Clear chat history
    if clear_button:
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Handle user input and generate assistant response
    if user_input or st.session_state.streaming:
        process_user_input(user_input)

def process_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        gpt_answer = chat(st.session_state.chat_history, 1000, model="gpt-3.5-turbo-16k", stream=True)
        st.session_state.generator = gpt_answer
        st.session_state.streaming = True
        st.session_state.chat_history.append({"role": "assistant", "content": ''})
        st.experimental_rerun()
    else:
        update_assistant_response()

def update_assistant_response():
    try:
        chunk = next(st.session_state.generator)
        st.session_state.chat_history[-1]["content"] += chunk
        st.experimental_rerun()
    except StopIteration:
        st.session_state.streaming = False
        st.experimental_rerun()

if __name__ == '__main__':
    main()