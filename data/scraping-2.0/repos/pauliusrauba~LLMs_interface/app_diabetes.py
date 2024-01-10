import os
from apikey import apikey
import pandas as pd
import numpy as np
import streamlit as st

from streamlit_chat import message as st_message
from streamlit.components.v1 import html

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from langchain.agents import initialize_agent
from langchain.agents import AgentType


from custom_tools import explain_predictions_diabetes, calculate_diabetes_risk, get_nice_guidelines, plot_feature_importance_heart_risk, df_to_string

# Set OpenAI api key
os.environ['OPENAI_API_KEY'] = apikey

# Prompt templates
template = """Assistant is an engaging, fun, verbose large language model which is an expert in medicine and machine learning, always asking for whether any additional analyses should be run.
Assistant is kind and provides a lot of information. Assistant always provides ALL information that it observes. 
Assistant always provides suggestions on what to evaluate next or what tool to use. Assistant evaluates cardiovascular risk very well using the newest methods.
The newest methods for cardiovascular risk predictions are the Qrisk scores. The Qrisk scores are the newest and most accurate methods for cardiovascular risk predictions.

Overall, Assistant is nice, always inquisitive and asks for clarifications on whether any additional analyses should be run. Assistant always provides ALL information that it observes. 
 """

@st.cache_resource()
def get_excel():
    df = st.session_state.df
    if df is not None:
        return df.to_excel()

@st.cache_resource()
def create_agent_chain():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='input', output_key="output")
    tools = [explain_predictions_diabetes, calculate_diabetes_risk, get_nice_guidelines]
    llm = ChatOpenAI(openai_api_key=apikey, temperature=0, max_tokens=3000, model_name='gpt-3.5-turbo') #model_name='gpt-4', gpt-3.5-turbo 
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                verbose=True, memory=memory, return_intermediate_steps=True,
                              input_key='input', output_key="output")
    
    agent_chain.agent.llm_chain.prompt.messages[0].prompt.template = template
    return agent_chain

tool_names = {
    'get_nice_guidelines': 'Get NICE guidelines',
    'calculate_diabetes_risk': 'Calculate diabetes risk',
    'explain_predictions_diabetes': 'Explain the prediction of diabetes',
}

def generate_answer():
    input_text = st.session_state.input_text
    agent_chain = create_agent_chain()
    response = agent_chain({"input":input_text})
    # Get tools used
    msg_response = response['output']
    if response['intermediate_steps'] != []:
        tool_used = response['intermediate_steps'][0][0].tool
        tool_input = response['intermediate_steps'][0][0].tool_input

        st.session_state.history.append({"message": input_text, "is_user": True})
        st.session_state.history.append({"message": msg_response, "is_user": False, "info": f"The model used the tool: {tool_names[tool_used]} with the following input: {tool_input}"})
    
        if tool_used == 'plot_feature_importance_heart_risk':
            # Import feature importance plot from feature_importance.txt
            with open('./cvd/feature_importance.txt', 'r') as f:
                html_img = f.read()

            st.session_state.history.append({"message": html_img, "is_user": False, "info": f"The model used the tool: {tool_used} with the following input: {tool_input}. The description of the tool is: Use this for any question related to plotting the feature importance of heart risk for any patient or any model. The input should always be an empty string and this function will always return a tuple that contains the top three risk and their associated scores. It will always plot of feature importances. "})            
        
        if tool_used == 'explain_predictions_diabetes':
            with open('./diabetes/feature_importance.txt', 'r') as f:
                html_img = f.read()
            st.session_state.history.append({"message": html_img, "is_user": False, "info": f"The model used the tool: {tool_used} with the following input: {tool_input}. "})     
    
    else:
        st.session_state.history.append({"message": input_text, "is_user": True})
        st.session_state.history.append({"message": msg_response, "is_user": False})

# Initialize streamlit history

def hide_code():
    st.markdown("""
        <style>
        code {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)
    
if "history" not in st.session_state:
    st.session_state.history = [{'message': 'Hello, I am the Medical AI assistant powered by the van der Schaar lab. How can I help you today?', 'is_user': False}]

st.title('🧠 💊 van der Schaar lab\'s AI Assistant')
st.text_input('Start the conversation with the Medical AI assistant below.', key='input_text', on_change=generate_answer)

if "df" not in st.session_state:
    st.session_state.df = None

file = st.file_uploader("Upload relevant patient data", type=['csv'])

if file:
    st.session_state.df = pd.read_csv(file)

for i, chat in enumerate(st.session_state.history):
    hide_code()

    # If info exists in the chat, display the message in two columns (one for the message, one for the info).
    if 'info' in chat:
        col1, col2 = st.columns([5, 1])  
        col2.write(st_message(chat['message'], is_user=chat['is_user'], allow_html=True, key=str(i))) 

        # Show more information when checkbox is checked
        if col1.checkbox('Show source information', key=f"info_btn_{i}"):
            st.info(chat['info'])
        else:
            st.write("")  # This line is need
    # Otherwise, display the message normally.
    else:
        st_message(chat['message'], is_user=chat['is_user'], allow_html=True, key=str(i))
