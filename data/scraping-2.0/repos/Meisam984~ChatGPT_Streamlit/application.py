import streamlit as st
from streamlit_chat import message
from PIL import Image
from decouple import config
import openai

openai.api_key = config("API_KEY")
image = Image.open("Logo.png")

st.set_page_config(page_title="Chatbot",
                   layout="centered")

st.image(image,
         use_column_width=True)

st.title("AI Consultant :red[ChatGPT] :sunglasses:")

if 'bot' not in st.session_state:
    st.session_state.bot = []

if 'user' not in st.session_state:
    st.session_state.user = []

def bot_response(query:str) -> str:
    response = openai.Completion.create(model="text-davinci-003",
                                        prompt=query,
                                        temperature=0.7,
                                        max_tokens=1024,
                                        stop=["\\n"],
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0
                                        )
    answer = response.choices[0].text
    return answer

def user_input() -> str:
    query = st.text_input(label="Me:",
                          value="Hi, How are you?",
                          key="input")
    return query

query = user_input()

if query:
    response = bot_response(query)
    st.session_state.user.append(query)
    st.session_state.bot.append(response)

if st.session_state.bot:
    for i in range(len(st.session_state.bot) - 1, -1, -1):
        message(message=st.session_state.bot[i],
                key=str(i)+"_bot_")
        message(st.session_state.user[i],
                is_user=True,
                key=str(i)+"_user_")
