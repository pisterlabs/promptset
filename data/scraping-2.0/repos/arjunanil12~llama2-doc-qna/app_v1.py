import streamlit as st
from dotenv import load_dotenv
from PIL import Image
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
# import PyPDF2
import os
from langchain.document_loaders import DirectoryLoader


from streamlit_option_menu import option_menu
import base64
from pathlib import Path
from sidebar import sidebar
from ingest import ingestMain


#==============================
# For conversion modules ######
#==============================

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message

from utils import *

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

#===============================

img= Image.open('EY.png')
save_folder=r"./source_documents"

def save_uploadedfile(uploadedfile,save_folder):
    with open(os.path.join(save_folder, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def main():
    load_dotenv()
    st.set_page_config(page_title="ET-GPT",page_icon=img, layout="wide")
    st.markdown("""
        <style>
        .big-font {
            font-size:70px !important;
            font-family: sans-serif;
            font-weight: bold;
            text-align: center
        }
        </style>
        """, unsafe_allow_html=True)
    

    col1,col2,col3= st.columns([1,1,50],gap="small")


    with col3:
        st.markdown('<p class="big-font">ET-GPT</p>', unsafe_allow_html=True)

    selected = option_menu(None, ["Home","Private LLM", "Public LLM", "Hybrid LLM", "Generic"], 
    icons=['house', 'incognito', "people-fill", 'toggles','list-ul'], 
    menu_icon="cast", default_index=0, orientation="horizontal",styles={
        # "container": {"padding": "0!important", "background-color": "#fafafa"},
        # "icon": {"color": "orange", "font-size": "25px"}, 
        # "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "yellow", "font-size": "18px","color": "black"},
    })
    
    if selected=="Home":
        homeCol1, homeCol2,homeCol3 =st.columns([6,100,1])
        with homeCol2:
            """
            ##### CUSTOM/PRIVATE LLM
            * Train your PDF’s using our framework to maximise accuracy.
            * Data is stored in secured cloud and works for Sensitive data as well.
            * Ideal for organization/team/process/financial services based data where  intelligent automation and Q&A can be achieved using LLMs.
            * *Eg: For an FS Client
            ##### HYBRID MODEL 
            * Sensitive data will be trained on Private LLM.
            * Open source/Cloud NLP /Open AI  services will be leveraged based on functionality to be solved and data sensitivity.
            * *Eg: Competitive Intelligence + Internal Process Automation
            ##### ADVANCED MODEL 
            * Can be directly connected to a pre-existing SQL system for Q&A​
            * Can be leveraged with ERP systems with Custom Data Migration layer​
            * Advisable for adopting Gen- AI in existing system ​
            * Data Privacy challenges to be addressed based on sensitivity​
            * *Eg: ERP connected – SCO client​
            ##### GENERIC  MODEL
            * Ideal for experiments and quick PoC’s where data privacy is not a challenge.​
            * Best proposed for showcasing capability and achievable with minimal subscription costs.​
            * *Eg: Tech transformation PoC – Sector Agnostic​

            """
    if selected=="Private LLM":
        choiseCo1, choiseCo2,choiseCo3 =st.columns([30,70,1])
        with choiseCo2:
            userChoise = st.radio("",options=["Query from existing knowledge base", "Ingest/Train new documents"],index=0,horizontal=True)
        if userChoise=="Ingest/Train new documents":
            modelCol1,modelCol2,modelCol3 =st.columns([6,100,100])
            # with modelCo1:
            #     modelOpt=st.selectbox(label="Select Model",options=["","GPT4All","Llama-7B","Llama-13B","Llama-70B"])
            #     if modelOpt:
            #         st.write(f"The Model opted for Private training is {modelOpt}")
            with modelCol2:
                uploaded_file = st.file_uploader("Upload your new training documents",type=["pdf", "doc", "txt","csv","ppt","pptx","html"],help="Scanned documents are not supported yet!", accept_multiple_files=True)
                for file in uploaded_file:
                    if uploaded_file is not None:
                        save_uploadedfile(file,save_folder)
                if st.button("Update knowledge base"):
                    with st.spinner("Processing"): 
                        ingestMain()
        if userChoise=="Query from existing knowledge base":
            userChoise1Col1,userChoise1Col2,userChoise1Col3 =st.columns([6,100,100])
            with userChoise1Col2:
                modelOpt=st.selectbox(label="Select Model",options=["","GPT4All","Llama-7B","Llama-13B","Llama-70B"])
                if modelOpt:
                    st.write(f"The Model opted for Private training is {modelOpt}")
                #if st.button("Start your conversation!",key='click'):
                #st.write("conversation started")
                
                # container for chat history
                response_container = st.container()
                # container for text box
                textcontainer = st.container()


                with textcontainer:
                    query = st.text_input("Query: ", key="input")
                    if query:
                        with st.spinner("typing..."):
                            conversation_string = get_conversation_string()
                            # st.code(conversation_string)
                            refined_query = query_refiner(conversation_string, query)
                            st.subheader("Refined Query:")
                            st.write(refined_query)
                            context = find_match(refined_query)
                            # print(context)  
                            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                        st.session_state.requests.append(query)
                        st.session_state.responses.append(response) 
                with response_container:
                    if st.session_state['responses']:

                        for i in range(len(st.session_state['responses'])):
                            message(st.session_state['responses'][i],key=str(i))
                            if i < len(st.session_state['requests']):
                                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')




    if selected=="Hybrid LLM":
        st.write("Hybrid")

    if selected=="Generic":
        sidebar()

    

if __name__ == '__main__':
    main()