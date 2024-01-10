import streamlit as st

import torch
from llama_index.embeddings import LangchainEmbedding
from llama_index import (GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
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



#openai.api_key = st.secrets.API_KEY


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

    # Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings
        
    params = {
            GenParams.MAX_NEW_TOKENS: 50, # The maximum number of tokens that the model can generate in a single run.
            GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
            GenParams.TEMPERATURE: 0.7,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
            GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation and avoid irrelevant tokens.
            GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative probability of at most P, helping to balance between diversity and quality of the generated text.
    }
        
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : st.secrets.WATSON_API
    }
    
    model_id = ModelTypes.LLAMA_2_70B_CHAT
    #model_id = "ibm/granite-13b-chat-v1"

  
        
    model = Model(
            model_id= model_id,
            credentials=credentials,
            params=params,
            project_id=st.secrets.PROJECT_ID)
    llm_hub = WatsonxLLM(model=model)
            #Initialize embeddings using a pre-trained model to represent the text data.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}        
    )
    
init_llm()


# LLMPredictor: to generate the text response (Completion)
llm_predictor = LLMPredictor(llm=llm_hub)
                                    
# Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
embed_model = LangchainEmbedding(embeddings)
 


#load documents

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
    # ServiceContext: to encapsulate the resources used to create indexes and run queries    
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            embed_model=embed_model
    )      
    # build index
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_data()


st.title("Chat with the Docs powered by IBM Watsonx.ai 💬🦙")
       
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "This is a fun app. It has information about PS & ITS  Team members! You can ask questions about them"}
    ]





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
