from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
openapi_key = st.secrets["OPENAI_API_KEY"]

# set streamlit page configuration
st.set_page_config(page_title="Hope to Skill ChatBot")
st.title("AI Mentor")

# initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = [] # store AI generated response

if 'past' not in st.session_state:
    st.session_state['past'] = [] # store past user input

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = "" # store the latest user input


# Initialize the open AI model
chat = ChatOpenAI(
    temperature= 0.5,
    model_name = "gpt-3.5-turbo",
    open_api_key= openapi_key,
)

def build_message_list():
    """
    Build a list of messages including system, human and AI messages
    """
    # start zipped_messages with the systemMessage
    zipped_messages = [SystemMessage(

        content = """Your name is AI Mentor. You are an AI Technical Expert for Artificial intelligence  """
    )]












# zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg # add user messages
            ))
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg) # add AI messages
            )
    return zipped_messages











def generate_response():
    """
    Generate AI response using the ChatOpenAi model.
    """
    # build the list of messages
    zipped_messages = build_message_list()

    # generate response using the chat model
    ai_response = chat(zipped_messages)

    response = ai_response.content

    return response










# create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt != "":
    # get user query
    user_query = st.secrets.entered_prompt

    # append user query
    st.session_state.past.append(user_query)

    # generate response
    output = generate_response()

    # append AI response to generated response
    st.session_state.generated.append(output)


# display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        # display AI response
        message(st.session_state["generated"][i], key = str(i))
        # display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')






