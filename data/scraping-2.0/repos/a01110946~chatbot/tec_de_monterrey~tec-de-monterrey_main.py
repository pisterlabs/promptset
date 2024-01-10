import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
import requests
import urllib.request
from PIL import Image

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/logo-tec.png', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/agent-v1.png', 'agent-image')
image2 = Image.open('agent-image')

# Streamlit UI.
st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:", layout="wide")
with st.container():  
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image,use_column_width='auto')#despliega logo
        st.header('Tec ChatBot')
        st.markdown("Many \"question-answering\" tools focus on unstructured text, yet much data is tabular. We've created a demo app to query table-based data using a dataset from Tecnológico de Monterrey's website. Take a look at the dataset [here](https://github.com/a01110946/chatbot/blob/main/tec_de_monterrey/Tecnologico-de-Monterrey_Curriculum.csv) so you know what type of questions you can ask!\n\nThis is not an official app and is not endorsed by Tecnológico de Monterrey.")
    with right_column:
        st.image(image2,use_column_width='auto')

st.sidebar.header('Hi, welcome!')
st.sidebar.markdown("""
The app's goal is to answer your questions regarding professional careers
and postgraduate courses offered by Tecnológico de Monterrey.
    
Ask questions to our Chatbot.
""")
st.title('Tec Chatbot')
st.info("TecChat Bot can provide answers to most of your questions regarding Tecnológico de Monterrey's curriculum.")

# GitHub file URL - Send a GET request to download the file - Save the file locally - Read CSV file and load Pandas DataFrame
file_url = "https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/Tecnologico-de-Monterrey_Curriculum.csv"
response = requests.get(file_url)
with open("Tecnologico-de-Monterrey_Curriculum.csv", "wb") as file:
    file.write(response.content)
df = pd.read_csv('Tecnologico-de-Monterrey_Curriculum.csv', encoding='ISO-8859-1')

# Initialize LLM and Pandas DataFram Agent using OpenAI Functions.
llm = ChatOpenAI(verbose=True, model="gpt-3.5-turbo-16k", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"], request_timeout=120, max_retries=2)
agent = create_pandas_dataframe_agent(llm, df, agent_type=AgentType.OPENAI_FUNCTIONS, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Which master's degrees are offered at Querétaro?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = agent({"input": prompt})
        full_response = response["output"]
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
