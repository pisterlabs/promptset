import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

import time

st.set_page_config(page_title="App built using watsonx.ai", layout="wide",initial_sidebar_state="auto", page_icon='👧🏻')
# load local CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles/styles_main.css")

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: lightblue;
    }
</style>
""", unsafe_allow_html=True)


# app sidebar
with st.sidebar:
    st.markdown("""
                # What can I ask ? 
                """)
    with st.expander("Click here to see FAQs"):
        st.info(
            f"""
            - Which city Anthony did his schooling ?
            - What is Anthonys favorite food ?
            - Which game Anthony, Vijay, Sudesh can play  ?
            - I want to visit a National Park. Who can advise me on it ?
            - I want to learn swimming. Whom should I speak to ?
          
            """
        )
    st.caption(f"Report bugs to sudesh@sg.ibm.com ")

with st.container():
    col1,col2 = st.columns([8,3])



openai.api_key = st.secrets.API_KEY


remove_all_streamlit_icons = """
  <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        .body {
        background-color: #f0f0f0; /* You can replace this with your desired color code */
        }
    </style>
"""


#st.markdown(remove_all_streamlit_icons, unsafe_allow_html=True)




st.title("Chat with the Docs powered by IBM Watsonx.ai 💬🦙")
       
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "This is a fun app. It has information about PS & ITS  Team members! You can ask questions about them"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Sudesh, Anthony, Jai, Ashoka, Naren, Manoj, Doyoung, Vijay are people who are part of a team. When someone ask for example who is Vijay, just provide answer from their profile. Answer any other questions based on the data provided. Assume that all questions are based on the information provided. Keep your answers short and  based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])



# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()  # Record the start time
            response = st.session_state.chat_engine.chat(prompt)
            elapsed_time = time.time() - start_time  # Calculate elapsed time

            # Check if elapsed time is less than 15 seconds
            if elapsed_time < 20:
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
            else:
                st.write("I am sorry, I dont want to keep you waiting. It's not something that I can answer within 15 seconds. Please ask the question in a different manner and I will try to answer within 15 seconds.")
                message = {"role": "assistant", "content": "Custom message for delayed response"}

            #st.session_state.messages.append(message)  # Add response to message history
