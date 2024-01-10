""" Main file for the chatbot. """
import os
from dotenv import load_dotenv
import json
load_dotenv()
import streamlit as st
from io import StringIO

from src.chat_agent import ChatAgent
from src.chroma_ import ChromaPy
from src.utils import chat_bot
from src.llama_index_exp import LlamaIndexExp
st.write("# Chat with your documents using langchain and streamlit")
openai_key = st.sidebar.text_input('OpenAI Key', '')


def pipeline(file_path:str) :
    def chroma_chat():
        ch = ChromaPy(openai_key)
        ch.prepare(txt_file=file_path)

        with st.expander("File content"):
            st.write(ch.raw_text)

        chat_bot = ChatAgent(fun=ch.chat_function)
        return chat_bot

    def llama_index_chat():
        li = LlamaIndexExp(openai_key)
        li.prepare()
        chat_bot = ChatAgent(fun=li.chat_function)
        return chat_bot

    chat_bot = llama_index_chat()
    return chat_bot

def upload_file():
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        string_data = bytes_data.decode("utf-8")
        file_path = os.path.join(os.getcwd(), 'my_file.txt')
        # Save the file in the current directory
        # with open(file_path, "w") as file:
        #     file.write(string_data)
        return file_path

def load_local_files():
    # load all text files in 'data' folder
    result ={}
    for file in os.listdir('data'):
        if file.endswith('.txt'):
            result[file] = {'name':os.path.splitext(file)[0] ,'path':os.path.join(os.getcwd(), 'data', file)}
    return result

def load_questions(chat_bot,file_name):

    # load question file from data
    question_file = os.path.join(os.getcwd(), 'data', 'data.json')
    with open(question_file) as json_file:
        questions = json.load(json_file)
    questions_list = ['']
    # get the question of file_name
    if file_name in questions.keys():
        questions_list.extend(questions[file_name])
    if len(questions_list)>1:
        st.sidebar.write("## Questions of the selected file")
        question = st.sidebar.selectbox('Select a question', questions_list)
        last_question = st.session_state.get('question', '')

        if len(question)>1 and last_question != question:
            chat_bot.external_question(question)
            st.session_state['question'] = question

local_files = load_local_files()
st.sidebar.write("## Local files")
file_path = st.sidebar.selectbox('Select a file', local_files.keys())
file_name = local_files[file_path]['name']
dd = local_files[file_path]['path']
chat_bot = pipeline(dd)
chat_bot.chat_bot()
load_questions(chat_bot,file_path)


# CMD LINE VERSION
#chat_bot(fun=ch.chat_function)

# WEB VERSION
