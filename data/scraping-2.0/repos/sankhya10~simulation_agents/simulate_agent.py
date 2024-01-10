from textwrap import indent
from urllib import response
import streamlit as st
import pandas as pd
#from turtle import onclick
import itertools

import pandas as pd
from streamlit_chat import message
from streamlit.components.v1 import html
import os
from langchain.chat_models import ChatOpenAI
from agent_calculation import *



st.set_page_config(
    page_title="Generative Agents: Interactive Simulacra of Human Behavior",
    page_icon="🤖",
    layout="wide",
)

with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n")
        api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        value=os.environ.get("OPENAI_API_KEY", None)
        or st.session_state.get("OPENAI_API_KEY", ""),
        )

if len(api_key_input) > 0:
    create_llm(api_key_input)
    

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.markdown(
        """ <style> .font {
        font-size:48px ; font-family: 'Arial'; color: black; text-align: centre;} 
            </style> """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="font">🤖 Creating Agents</p>',
        unsafe_allow_html=True,
    )
    st.header("")


if "open_api_key" not in st.session_state:
    st.session_state.open_api_key = api_key_input

if "agents" not in st.session_state:
    st.session_state.agents = []

if "name" not in st.session_state:
    st.session_state.name = []

if "age" not in st.session_state:
    st.session_state.age = []

if "gender" not in st.session_state:
    st.session_state.gender = []

if "traits" not in st.session_state:
    st.session_state.traits = []

if "status" not in st.session_state:
    st.session_state.status = []  

if 'check_uploads' not in st.session_state:
    st.session_state.check_uploads = []

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = False 


if "agent_index" not in st.session_state:
    st.session_state.agent_index = 0

if "user_name" not in st.session_state:
    st.session_state.user_name = []


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


if "finalize_upload" not in st.session_state:
    st.session_state.finalize_upload = False


if "interview_agent" not in st.session_state:
    st.session_state.interview_agent = ""

if "interview" not in st.session_state:
    st.session_state.interview = False


if "display_column" not in st.session_state:
    st.session_state.display_column = False


if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = None


if "finalize_memory" not in st.session_state:
    st.session_state.finalize_memory= False


if "input_query" not in st.session_state:
    st.session_state.input_query= []



def get_text(user_name,agent_selection):

    c35, c36, c37  = st.columns([1.2,8,1])
    with c35:
        st.text("")
        st.text("")
        st.text("")
        st.markdown(f"##### Insert Query :red[{user_name}]:")
    with c36:
        st.text("")
        input_query.append(st.text_input("", key=10, value= ""))
    with c37:
        st.text("")
        st.text("")
        m = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                    background-color: rgb(200, 175, 0);height:2.75em;width:6.2em;color:black;font-size:15px;margin-top:-3.8px}
                    </style>""", unsafe_allow_html=True)
        send = st.button("Send",on_click = clear_query)
        st.text("")
        if send and st.session_state.input_query[0] != "":
            return st.session_state.input_query[0]





def add_agent_details(name,age,gender, traits, status, memories):

    agent = {
        'name': name,
        'age': age,
        'gender':gender,
        'Traits': traits,
        'status': status,
        'initial_memories': memories}
    
    st.session_state.agents.append(agent)


def clear_query():
    st.session_state.input_query=input_query
    st.session_state[str(10)] = ""




def clear():
    st.session_state.name = name
    st.session_state.age = age
    st.session_state.gender = gender
    st.session_state.traits = traits
    st.session_state.status = status
    st.session_state[str(1)]=""
    st.session_state[str(2)]= 18
    st.session_state[str(3)]=""
    st.session_state[str(4)]=""
    st.session_state[str(5)]=""



if __name__ == "__main__":
    name = []
    age = []
    gender = []
    traits = []
    status = []
    user_name= []
    input_query=[]

    st.markdown("### Add agent details")

    st.text("")

    col1, col2, col3, col4, col5  = st.columns(5)

    # Add input fields to each column
    with col1:
        name.append(st.text_input("Add Agent's Name",key=1))
    with col2:
        age.append(st.number_input("Input Agent's Age",min_value=18,max_value=100,key=2))
    with col3:
        gender.append(st.text_input("Input Agent's Gender",key=3))
    with col4:
        traits.append(st.text_input("Input Agent's Traits",key=4))
    with col5:
        status.append(st.text_input("Input Agent's Status",key=5))

    st.text("")
    st.text("")
    

    c38, c39 = st.columns([5,5])
    with c38:
        st.text("")
        st.text("")
        st.markdown("###### Add memories via uploading a file :open_file_folder: ")
        uploaded_file = st.file_uploader(label="",accept_multiple_files=False,key=6)
    

    if uploaded_file:
        uploaded_file.seek(0)
        sentences = uploaded_file.read().splitlines()
        if st.session_state.uploaded_file != True:
            if name and age and gender and traits and status:
            # Add agent details to the list
                with c39:
                    if uploaded_file not in st.session_state.check_uploads:
                        st.text("")
                        st.text("")
                        output_list = [item.decode('utf-8') for item in sentences]
                        st.markdown(f"###### Memories for :blue[{name[0]}]")
                        st.session_state.memory_state = True
                        txt = st.text_area("",value = ("\n").join(output_list),height=170)
                        m = st.markdown("""
                            <style>
                            div.stButton > button {
                            background-color:white;height:3.75em;width:13em;color:black;font-size:15px
                                }
                            </style>""", unsafe_allow_html=True)
                        memory_finalized = st.button("Finalize Memory")
                        if memory_finalized:
                            changed_list = txt.split('\n')
                            add_agent_details(name, age, gender, traits, status, changed_list)
                            st.session_state.check_uploads.append(uploaded_file)
                            st.session_state.finalize_memory = True
                
            if st.session_state.finalize_memory:
                m = st.markdown("""
                            <style>
                            div.stButton > button {
                            background-color:white;height:2.75em;width:30.5em;color:black;font-size:15px
                                }
                            </style>""", unsafe_allow_html=True)
                final = st.button("Finalize Agent",on_click=clear)
                if final:
                    st.session_state.agent_index += 1
                    if st.session_state.agent_index == 2:
                        with st.spinner("Generating initial summary ..."):
                            st.session_state.agent_memory, summary = create_initial_summary(st.session_state.agents,api_key_input)
                            file_container = st.expander("Check your Agent Summaries")
                            file_container.write(summary)
                        st.session_state.display_column = True
                    
        if st.session_state.display_column:
            st.session_state.finalize_memory = False
            c30, c31, c32 = st.columns([2.5, 1, 3])

            with c30:
                # st.image("logo.png", width=400)
                st.markdown(
                    """ <style> .font {
                    font-size:48px ; font-family: 'Arial'; color: black; text-align: centre;} 
                        </style> """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="font">💬 Interviewing Agents </p>',
                    unsafe_allow_html=True,
                )
                st.header("")

            c33, c34 = st.columns([4,4])

            with c33:

                st.markdown("#### :red[Enter interviewer details] :male-student: ")
                st.session_state.interview = True
            
                if  st.session_state.interview:
                    st.session_state.uploaded_file = False
                    user_name.append(st.text_input("",value = "",key=7))
                
            with c34:
                st.markdown("#### :blue[Select the interviewee agent] :robot_face: ")
                agent_names = list(st.session_state.agent_memory.keys())
                options = ["Select Agent", *agent_names]
                st.session_state.interview_agent = st.selectbox("",tuple(options),index=0)   
            

            if len(user_name[0]) > 0 and st.session_state.interview_agent != "Select Agent":
                st.session_state.display_column = True

                with st.expander("Chat Now!"):
                    placeholder = st.empty()
                    user_input = get_text(user_name[0],st.session_state.interview_agent)
                    new_message = f"{user_name[0]} says {user_input}"
                    
                    if user_input != None:
                        st.session_state.display_column = True
                        tommy = st.session_state.agent_memory[st.session_state.interview_agent]
                        with st.spinner("Generating response ..."):
                            _ , output = tommy.generate_dialogue_response(new_message)      
                            st.session_state['past'].append(new_message)
                            st.session_state.generated.append(output)
                    
                    if st.session_state['generated']: 
                        st.session_state.display_column = True
                        with placeholder.container():
                            for i in range(len(st.session_state['generated'])):
                                message(st.session_state['past'][i], is_user=True, key=str(i+8) + '_user')
                                message(st.session_state["generated"][i], key=str(i+8))

                    css='''
                        <style>
                        [data-testid="stExpander"] div:has(>.streamlit-expanderContent) {
                        overflow: scroll;
                        height: 700px;
                        }
                        </style>
                        '''
                    st.markdown(css, unsafe_allow_html=True)
