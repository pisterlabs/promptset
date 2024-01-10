import os
import streamlit as st
import openai

from streamlit_chat import message as st_message

st.set_page_config(
    page_title = 'Streamlit Sample Dashboard Template',
    page_icon = '✅',
    layout = 'wide'
)


    

# api_key = os.environ.get('chagpt_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

if "history_1" not in st.session_state:
    st.session_state.history_1 = []
    
if "history_2" not in st.session_state:
    st.session_state.history_2 = []
    
# if "input_text" not in st.session_state:
#     st.session_state.input_text = ""
    



def get_model_reply(query, context=[]):
    # combines the new question with a previous context
    context += [query]
    
    # given the most recent context (4096 characters)
    # continue the text up to 2048 tokens ~ 8192 charaters
    completion = openai.Completion.create(
        engine='text-davinci-003', # one of the most capable models available
        prompt='\n\n'.join(context)[:4096],
        max_tokens = 100,
        temperature = 0.0, # Lower values make the response more deterministic
    )
    
    # append response to context
    response = completion.choices[0].text.strip('\n')
    context += [response]
    
    # list of (user, bot) responses. We will use this format later
    responses = [(u,b) for u,b in zip(context[::2], context[1::2])]

    
    return responses, context


def generate_answer_1():
    user_message = st.session_state.input_text_1
    responses, context = get_model_reply(user_message)
    bot_message = responses[-1][1]
    st.session_state.history_1.append({"message": user_message, "is_user": True})
    st.session_state.history_1.append({"message": bot_message, "is_user": False})


def generate_answer_2():
    user_message = st.session_state.input_text_2
    responses, context = get_model_reply(user_message)
    bot_message = responses[-1][1]
    st.session_state.history_2.append({"message": user_message, "is_user": True})
    st.session_state.history_2.append({"message": bot_message, "is_user": False})


botscreen1, botscreen2 = st.columns([2,2])
with botscreen1:
    st.title('BOT 1')
    user_input_1 = st.text_input("What is on your mind 1?",
        key = 'input_text_1',
        on_change=generate_answer_1)
    
    for chat in st.session_state.history_1:
        st_message(**chat)  # unpacking
    
    st.write(st.session_state.history_1)

    
with botscreen2:
    st.title('BOT 2')
    user_input_2 = st.text_input("What is on your mind 2?",
        key = 'input_text_2',
        on_change=generate_answer_2)
    for chat in st.session_state.history_2:
        st_message(**chat)  # unpacking
        

    st.write(st.session_state.history_)
