import streamlit as st
import openai
import time 

from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex,SimpleDirectoryReader,Document

from llama_index import StorageContext,load_index_from_storage,ServiceContext,set_global_handler


st.set_page_config(page_title="KYC Application", layout="wide",initial_sidebar_state="auto", page_icon='👧🏻')
# app sidebar
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
st.title("Chat application built with Streamlit docs, powered by LlamaIndex")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Know your Colleagues (KYC), Ask about your colleagues "}
    
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        prompt='''
        You are an expert on people. Your job is to answer  questions about people. 
        Assume that all questions are related to data provided. 
        Keep your answers short and based on facts – do not hallucinate features.
        
        Question: Who is Vijay ?
        Answer: Vijay has 24 years’ experience in Enterprise Infrastructure Design, Solution Architecture and Cloud Solution & Integration. Vijay has a strong Technical Project Management, Transformation and Transition of customer Data center and Team Management experience. Vijay specialities are Hyperscalers like AWS, Azure, Google and IBM cloud, EMC, IBM & VMware products, Management & Orchestration for Cloud Environment, Virtualization Architecture, Data center Consolidation, Data center Management, Technical presales, BC & DR setup, Backup. Setting up Proof of Concepts to demonstrate the product capability, Team Management & Team Player and evolving every day in the ever-changing world.
        Question: Who is Sudesh ?
        Answer: Sudesh has worked in Software for over 22 . Sudesh has a diverse set of skillsets serving in technical roles across consulting, presales, support & services. Sudesh is also a hands-on Architect, and has worked across ASEAN, India  and Australia. 
        '''
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", chunk_size=150,chunk_overlap=0,temperature=0.3,max_token=50,system_prompt=prompt))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        #index.storage_context.persist()
        #storage_context = StorageContext.from_defaults(persist_dir="./storage")
        #index = load_index_from_storage(storage_context=storage_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
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





