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
import shutil


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


import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from utils import *
from langchain.chains.question_answering import load_qa_chain

import pandas as pd

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt4", openai_api_key="ea573133f4cb49ef900a596d0af317a3")

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


import streamlit as st
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader


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


    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login('Login', 'main')

    if st.session_state["authentication_status"]:
        st.write(f'Welcome *{st.session_state["name"]}*')

        # reading the CSV file
        df_persona = pd.read_excel('configX.xlsx', 'persona')
        
        if st.session_state["name"] :
            persona = df_persona.loc[df_persona['user'] == username, 'persona'].iloc[0]

    # Inside Authentication - If Success
        col1,col2,col3= st.columns([1,1,50],gap="small")


        with col3:
            st.markdown('<p class="big-font">AM&M</p>', unsafe_allow_html=True)

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
                **Generative AI (gen-AI)** holds the potential to reshape the business landscape for AM&M (Additive Manufacturing & Materials) companies in profound ways. By harnessing the power of gen-AI, these companies can reimagine their customer relationship model, ushering in a new era of personalized engagement and satisfaction.
                The integration of generative AI enables AM&M companies to tailor their interactions with both direct purchasers and end consumers, reaching new heights of customer satisfaction. This technology facilitates real-time customization, allowing for bespoke solutions and post-sale services that cater to individual preferences and requirements.
                Beyond conventional boundaries, gen-AI empowers AM&M companies to transcend traditional product offerings. This strategic shift involves the creation of innovative smart connected products and services, amplifying revenue streams and catalysing growth.
                """

                image = Image.open('homescreenAMM.png')
                st.image(image)#, caption='Sunrise by the mountains')

                """
                These offerings may seamlessly integrate IoT capabilities, data-driven insights, and cutting-edge technologies to deliver unmatched value.
                Moreover, the transformative potential of gen-AI extends to crafting immersive experiences that seamlessly complement existing products. Through the synergy of virtual reality, augmented reality, and other immersive technologies, AM&M companies can pioneer novel ways of engagement, setting themselves apart as pioneers of innovation.
                A key facet of leveraging gen-AI lies in forging robust collaborations within partner ecosystems.
                """ 

                image_1 = Image.open('homescreenAMM_1.png')
                st.image(image_1)# caption='Sunrise by the mountains')
                
                """
                By tapping into external expertise, AM&M companies can unlock a realm of possibilities. Collaborative efforts can result in ground-breaking products, services, and market strategies that transcend individual capabilities.
                Embracing generative AI necessitates a strategic approach, encompassing research, development, and seamless integration into established business workflows. Concomitantly, ethical considerations, data privacy, and security must remain paramount in this transformative journey.
                In essence, your insights underscore the profound impact of generative AI on the AM&M sector. By embracing this paradigm shift, companies can usher in an era of personalized engagement, innovation, and strategic collaboration, revolutionizing the way business is conducted and positioning themselves as leaders in a dynamic market landscape.


                ##### How we can help?
                ##### 1. Generative AI Strategy and Roadmap
                * Build an AI Strategy
                * Benefits Realization/ Adoption Framework with Information Security
                * Business Case expansion and Ideation
                * Target Operating Model Design based on strategic approaches
                * Roadmap & Execution Plan

                ##### 2. Data Security and Responsible AI
                * AI/ML Data Handling, Privacy and Security
                * Tailor made solution build with model Change Management

                ##### 3. Scalability and Reliability
                * Process Design improvement
                * Technology/Infrastructure Build
                * Interoperabile architecture 
                * Multi-channel Integration support

                ##### Our 4 main approaches are : Private LLM | Hybrid LLM | Advanced LLM | Generic LLM
                

                """
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
            choiseCo1, choiseCo2,choiseCo3 =st.columns([20,90,1])
            with choiseCo2:
                if persona == "admin" :
                    userChoise = st.radio("",options=["Query from existing knowledge base", "Ingest/Train new documents","Knowledge Base", "Configuration"],index=0,horizontal=True)
                else :
                    userChoise = st.radio("",options=["Query from existing knowledge base"],index=0,horizontal=True)
            
            
            if userChoise=="Ingest/Train new documents":

                modelCol1,modelCol2,modelCol3 =st.columns([6,100,100])
                
                # with modelCo1:
                #     modelOpt=st.selectbox(label="Select Model",options=["","GPT4All","Llama-7B","Llama-13B","Llama-70B"])
                #     if modelOpt:
                #         st.write(f"The Model opted for Private training is {modelOpt}")
                with modelCol2:
                    uploaded_file = st.file_uploader("Upload your new training documents",type=["pdf", "doc", "txt","csv","ppt","pptx","html"],help="Scanned documents are not supported yet!", accept_multiple_files=True)
                    
                    filelist=[]
                    for root, dirs, files in os.walk("./source_documents"):
                        for file in files:
                                filename=os.path.join(root, file)
                                filelist.append(filename.split("/")[-1])

                    
                    
                    for fileCount,file in enumerate(uploaded_file):
                        if uploaded_file is not None:
                            # st.write(uploaded_file[fileCount].name)
                            inputSelect=st.multiselect(f'Map the input file **{uploaded_file[fileCount].name}** with exisiting knowledge base documents!',filelist)
                            if inputSelect:
                                catName=st.text_input("Category Name:",key="newinput_"+str(fileCount))
                                if st.button('Create Mapping',key="newbutton_"+str(fileCount)):
                                    if catName:
                                        try:
                                            #os.mkdir(catName)
                                            os.makedirs("Mapping/"+catName)
                                            save_uploadedfile(file,"Mapping/"+catName)
                                            #save_uploadedfile(file,catName)
                                            for inputSelectFile in inputSelect:
                                                shutil.copy(f"./source_documents/{inputSelectFile}",  "./Mapping/"+catName)
                                                #shutil.copy(f"./source_documents/{inputSelectFile}", f"./{catName}")
                                            st.write(f"Successfully created the mapping:- **{catName}**")
                                        except:
                                            save_uploadedfile(file,"Mapping/"+catName)
                                            for inputSelectFile in inputSelect:
                                                shutil.copy(f"./source_documents/{inputSelectFile}",  "./Mapping/"+catName)
                                                #shutil.copy(f"./source_documents/{inputSelectFile}", f"./{catName}")
                                            st.write(f"Successfully created the mapping:- **{catName}**")
                                    catName = ''
                                    

                    if st.button("Update knowledge base"):
                        for file in uploaded_file:
                            if uploaded_file is not None:
                                save_uploadedfile(file,save_folder)
                        with st.spinner("Processing"): 
                            ingestMain()

                # chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")
                # query = "How is the annual report of 2022 of carrier?"
                # chain.run(input_documents=documents, question=query)


            if userChoise=="Query from existing knowledge base":
                refined_query=""
                userChoise1Col1,userChoise1Col2,userChoise1Col3 =st.columns([6,100,100])
                with userChoise1Col2:
                    modelOpt=st.selectbox(label="Select Model",options=["","GPT4All","Llama-7B","Llama-13B","Llama-70B"])
                    if modelOpt:
                        st.write(f"The Model opted for Private training is {modelOpt}")
                    conVCol1,conVCol2,conVCol3 =st.columns(3)
                    chatCol1,chatCol2=st.columns([100,1])
                    with conVCol1:
                        # if st.button("Start your conversation!",key='click'):
                        #     st.write("conversation started")
                            
                            
                        with chatCol1:
                            # container for chat history
                            response_container = st.container()
                            # container for text box
                            textcontainer = st.container()
                            #st.session_state.text = ""

                            with textcontainer:
                                query= st.text_input('Queries:')
                                if query!="":
                                    filtered_query = query_filter_userbased(query, persona)
                                    #st.subheader("User Restriction:")
                                    #st.write(filtered_query)
                                    if filtered_query=='Yes':
                                            st.session_state.requests.append(query)
                                            st.session_state.responses.append("You don't have access to this data! Kindly proceed with another Query.")

                                    else:
                                        with st.spinner("typing..."):
                                            conversation_string = get_conversation_string()
                                            # st.code(conversation_string)

                                            refined_query = query_refiner(conversation_string, query)
                                            #t.subheader("Refined Query:")
                                            print(refined_query)

                                            context = find_match(refined_query)
                                            # print(context)  
                                            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                                        st.session_state.requests.append(query)
                                        st.session_state.responses.append(response) 
                            with response_container:
                                if st.session_state['responses']:

                                    for i in range(len(st.session_state['responses'])):
                                        message(st.session_state['responses'][i],key=str(i))
                                        # if refined_query!="":
                                            # st.markdown(f'_Refined_ _Query_ :**{refined_query}**')
                                        if i < len(st.session_state['requests']):
                                            message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
                                            if refined_query!="":
                                                st.markdown(f"_Refined_ _Query_ :{refined_query}")
                                            

                    with conVCol1:
                        if st.button("Clear conversation!"):
                            st.write("past conversation cleared")
                            # Delete all the items in Session state
                            for key in st.session_state.keys():
                                del st.session_state[key]
            
            
            
            if userChoise=="Knowledge Base":
                filelist=[]
                for root, dirs, files in os.walk("./source_documents"):
                    for file in files:
                            filename=os.path.join(root, file)
                            filelist.append(filename.split("/")[-1])
                for fileName in filelist:
                    with st.expander(fileName):
                        st.write("File")

                st.header("Mapped Document Information")
                mappedfilelist=[]
                dirsFilelist=[]
                
                for root, dirs, files in os.walk("./Mapping"):
                    # st.write(root)
                    # st.write(os.listdir(root))
                    for dir in dirs:
                        dirsFilelist.append(dir)

                        for file in files:
                                filename=os.path.join(root, file)
                                mappedfilelist.append(filename.split("/")[-1])

                # st.write(mappedfilelist)
                for dirName in dirsFilelist:
                    with st.expander(dirName):
                        # st.write("files")
                        #filelistval=st.write(os.listdir("./Mapping/"+dirName))
                        filelistval=os.listdir("./Mapping/"+dirName)
                        for mappedfilelistname in filelistval:
                            st.write(mappedfilelistname)

            
            if userChoise=="Configuration":
                df_rules = configuration_layer()
                with pd.ExcelWriter('rules.xlsx') as writer :
                    # write dataframe to excel
                    df_rules.to_excel(writer, sheet_name = 'rules')
                            
                                
        if selected=="Public LLM":
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
                # container for chat history
                response_container = st.container()
                # container for text box
                textcontainer = st.container()


                with textcontainer:
                    query = st.text_input("Query: ")#, key="input")
                    #filtered_query = persona_based_qparser(query)
                    filtered_query = query_filter_userbased(query, persona)
                    if query:
                        with st.spinner("typing..."):
                            conversation_string = get_conversation_string()
                            print(query)
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




        authenticator.logout('Logout', 'main', key='unique_key2')


    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
    



def configuration_layer() :
    
    df_rules = pd.read_excel('rules.xlsx', 'rules')
    col1, col2, col3 = st.columns([10,100,100])
    rule_list = ['Hide Financial Values', 'Hide Sensitive Names']

    selection_1 = df_rules['admin'].tolist()
    selection_2 = df_rules['supplier'].tolist()

    selection_1 = [i for i in selection_1 if i in rule_list]
    selection_2 = [i for i in selection_2 if i in rule_list]

    with col2:
        st.header("Admin")    
        option_1 = st.multiselect('Rule Selection',
                    rule_list, selection_1, key = "ukey1")

    with col3:
        st.header("Supplier")
        option_2 = st.multiselect('Rule Selection',
                    rule_list, selection_2, key = "ukey2")
    
    
    df_rules['admin'] = None
    t_option_1 = option_1
    if len(option_1) < 2 :
        t_option_1.extend(['x']* (2-len(option_1)))
    df_rules['admin'] = t_option_1

    df_rules['supplier'] = None
    t_option_2 = option_2
    if len(option_2) < 2 :
        t_option_2.extend(['x']* (2-len(option_2)))
    df_rules['supplier'] = t_option_2

    return df_rules

if __name__ == '__main__':
    main()