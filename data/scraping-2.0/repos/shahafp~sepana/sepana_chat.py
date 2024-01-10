import re

import streamlit as st
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

import auto_user
import utils

st.title('🦜🔗 Sepana Home Assigment')

with st.sidebar:
    st.title('🤗💬 SepanaChat App')
    add_vertical_space(5)
    st.write('Made with ❤️ by Shahaf Pariente')
    openai_api_key = st.sidebar.text_input('OpenAI API Key')

    st.session_state.auto_user_button = st.button("Auto User :rocket:")
    st.write("Start auto user")
    st.write("Train your user before hit the Auto User button")
    st.write(st.session_state.auto_user_button)

    if st.session_state.auto_user_button:
        st.session_state.button_state = True
        st.session_state.auto_user_conv.memory.chat_memory.add_user_message("This is the end of the Human examples, "
                                                                            "pay attention when the Human ended the chats, now it is your time to shine and answer like you are the Human")

# Generate empty lists for generated and past.
# generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm SepanaChat, give me your scenario please :)"]
# past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hi!"]

if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Layout of input/response containers
# input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

if openai_api_key.startswith('sk-'):
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0.4,
                     openai_api_key=openai_api_key,
                     model_name='gpt-3.5-turbo',
                     verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
        entity_memory = ConversationBufferMemory(llm=llm)
        entity_memory.chat_memory.add_user_message(utils.SCENARIO_EXAMPLE)
        entity_memory.chat_memory.add_ai_message(utils.FIRST_OPTIONS)
        entity_memory.chat_memory.add_user_message(utils.USER_FIRST_CHOICE)
        entity_memory.chat_memory.add_ai_message(utils.SECOND_OPTIONS)
        entity_memory.chat_memory.add_user_message(utils.USER_SECOND_CHOICE)
        entity_memory.chat_memory.add_ai_message(utils.FINAL_OPTIONS)
        entity_memory.chat_memory.add_user_message(utils.USER_THIRD_CHOICE)
        st.session_state.entity_memory = entity_memory

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=utils.prompt,
        memory=st.session_state.entity_memory,
    )

    auto_user_conv = auto_user.get_conversation_chain(llm)
    st.session_state.auto_user_conv = auto_user_conv
else:
    st.markdown(''' 
        ```
        - 1. Enter API Key + Hit enter 🔐 

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```

        ''')
    # st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')


# Function for taking user provided prompt as input
def get_text():
    # input_text = st.text_input("You: ", "", key="input")
    input_text = st.chat_input()
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if input_text and openai_api_key.startswith('sk-'):
        return input_text


# Applying the user input box
with st.container():
    user_input = get_text()


# Response output
# Function for taking user prompt as input followed by producing AI generated responses
def generate_response(input_text):
    return Conversation.run(input=input_text)


def generate_auto_user(input_text):
    return st.session_state.auto_user_conv.run(input=input_text)


# Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input and 'button_state' not in st.session_state:
        response = generate_response(user_input) if not re.search(r"\bend\b", user_input, re.IGNORECASE) else 'Thank you!'
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        st.session_state.auto_user_conv.memory.chat_memory.add_user_message(user_input)
        st.session_state.auto_user_conv.memory.chat_memory.add_ai_message(response)

    elif user_input and st.session_state.button_state:
        # To run the Conversational between 2 bots we need a loop that iterate over the responses
        st.session_state.index = 1
        return_val = auto_user.auto_user_loop(st=st, user_input=user_input, generate_response=generate_response, generate_auto_user=generate_auto_user)
        st.session_state.generated.append(return_val)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
