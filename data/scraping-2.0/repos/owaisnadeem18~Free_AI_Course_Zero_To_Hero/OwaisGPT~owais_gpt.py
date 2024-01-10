from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    AIMessage,
    HumanMessage
)
chat_api = st.secrets["OPENAI_API_KEY"]

# Set Page configuration in your chat-bot (Owais-AI-GPT)

st.set_page_config(page_title='Owais-AI-GPT')
st.title("Owais-AI-GPT")

# Initiaile the session state variables:
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""

# Give access to Chat GPT:

chat = ChatOpenAI(

    model_name='gpt-3.5-turbo',
    temperature=0.5,
    openai_api_key=chat_api,
    max_tokens=100
)


def all_messages():
    """
        Build a list of messages including AI Message , human and system messages
    """

    zipped_messages = [SystemMessage(

        content='''

            your name is Owais-AI-GPT . You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and an professional tone.

                1. Greet the user politely and professionally , politely ask user name and ask how you can assist them with AI-related queries.
                2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
                3. you must Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
                5. Be patient and considerate when responding to user queries, and provide clear explanations.
                6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                7. Do Not generate the long paragarphs in response. Maximum Words should be 100.

                Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being.

'''

    )]

    # zip together past and generated messages

    for human_message, ai_message in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_message is not None:
            zipped_messages.append(HumanMessage(content="human_message"))

        if ai_message is not None:
            zipped_messages.append(AIMessage(content="ai_msg"))

    return zipped_messages


def generate():
    """
            Generate AI response using the ChatOpenAI model.
    """

    # Build the list of messages:
    zipped_messages = all_messages()

    # Generate response using chat model:
    ai_response = chat(zipped_messages)

    return ai_response.content

# Define function to submit user input:


def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input

    # Clear prompt Input
    st.session_state.prompt_input = ""


# streamlit UI Search box:
st.text_input("Ask anything about AI from Owais-AI-GPT",
              key="prompt_input", on_change=submit)

if st.session_state.entered_prompt != "":
    # Get User Query
    user_question = st.session_state.entered_prompt

    # Append User query to past query
    st.session_state.past.append(user_question)

    # Generate Response
    output = generate()

    # Append AI response to generated responses:
    st.session_state.generated.append(output)

# Display the chat history:
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI Response:
        message(st.session_state['generated'][i], key=str(i))
        # Display User Message:
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')

# -------------------------Completed---------------------------X
