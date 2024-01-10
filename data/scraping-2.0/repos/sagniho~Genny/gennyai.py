import streamlit as st
import openai
from openai import OpenAI
import base64

st.set_page_config(page_title="Genny AI Website Advisor", page_icon="gen.png")
# Set your OpenAI API key
# Access API key from environment variable
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# Use the Assistant ID from the Assistant you created
ASSISTANT_ID = st.secrets['GENNY_ASSISTANT_ID_v2']
with st.sidebar:
    st.image("bench.png", use_column_width="auto")
    st.subheader("Welcome to [Benchmark Gensuite®.](https://benchmarkgensuite.com) Unified, organically developed, and integrated digital solutions for EHS, sustainability, quality, operational risk, product stewardship, supply chain, and ESG disclosure reporting.  ")

# Create columns for the logo and the title
col1, col2 = st.columns([1, 4])

# In the first column, display the logo
with col1:
    st.image('genny.png', width=175)  # Adjust the width as needed

# In the second column, display the title and subtitle
with col2:
    st.markdown("<h2 style='margin-top: 0;'>Benchmark Gensuite® Product Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top: 0; padding-left: 5px; color: green; font-style: italic;'>Your interactive Genny AI-powered guide to the Benchmark Gensuite solutions platform</p>", unsafe_allow_html=True)


def send_message_get_response(assistant_id, user_message):
    # Create a new thread
    thread = client.beta.threads.create()

    # Add user message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    # Retrieve the assistant's response
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            latest_message = messages.data[0]
            text = latest_message.content[0].text.value
            return text


def main():
    # Initialize messages in session state if not present
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display previous chat messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="genn.png"):
                st.write(msg["content"])
    
    # Quick ask buttons setup
    if 'quick_ask_shown' not in st.session_state:
        st.session_state['quick_ask_shown'] = True

    # Initialize 'quick_ask_flag' in session state if not present
    if 'quick_ask_flag' not in st.session_state:
        st.session_state['quick_ask_flag'] = 0  # 0 means no quick ask button has been clicked

     # Initialize 'quick_ask_flag' in session state if not present
    if 'quick_ask_q' not in st.session_state:
        st.session_state['quick_ask_q'] = ''  # 0 means no quick ask button has been clicked

    # Placeholder for quick ask buttons, which will be at the bottom above chat input
    quick_ask_placeholder = st.empty()

    # Display quick ask buttons above the chat input
    if st.session_state['quick_ask_shown']:
        with quick_ask_placeholder.container():
            st.write("\n\n\n\n\n")
            st.write("\n\n\n\n\n")
            st.write("\n\n\n\n\n")
            st.write("Some questions you can ask me…whatever you want to know about Benchmark Gensuite, just ask!")
            quick_asks = [
                "How do I get started?",
                "What should I implement?",
                "Why select Benchmark Gensuite?"
            ]
            # Use columns to display quick asks in a row
            cols = st.columns(len(quick_asks), gap="small")
            for col, ask in zip(cols, quick_asks):
                with col:
                    if st.button(ask):
                        st.session_state['quick_ask_q'] = ask # Pre-populate chat input with quick ask
                        st.session_state['quick_ask_shown'] = False  # Hide quick asks after use
                        st.session_state['quick_ask_flag'] = 1
            
            if st.session_state['quick_ask_flag'] == 1:          
                quick_ask_placeholder.empty()   

    if st.session_state['quick_ask_flag'] == 1:          
        process_user_input(st.session_state['quick_ask_q'])

    # Chat input for new message
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ''  # Initialize user_input in session state

    user_input = st.chat_input(placeholder="Please ask me your question…")

    # When a message is sent through the chat input or a quick ask button is clicked
    if user_input:
        quick_ask_placeholder.empty()  # Remove quick ask buttons from the layout
        process_user_input(user_input)
        st.session_state['user_input'] = ''  # Clear chat input after processing

def process_user_input(user_input):
    # Append the user message to the session state
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    # Display the user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)

    # Get the response from the assistant
    with st.spinner('Working on this for you now...'):
        response = send_message_get_response(ASSISTANT_ID, user_input)
        # Append the response to the session state
        st.session_state['messages'].append({'role': 'assistant', 'content': response})
        # Display the assistant's response
        with st.chat_message("assistant", avatar="genn.png"):
            st.write(response)
    



if __name__ == "__main__":
    main()