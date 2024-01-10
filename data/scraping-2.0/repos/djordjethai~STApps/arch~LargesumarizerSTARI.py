# This code sets up the necessary components, interacts with the LangChain tool and ChatOpenAI model to perform text summarization, 
# and provides a user interface for input and output.

from langchain.document_loaders import UnstructuredFileLoader  # Importing necessary modules
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate  # Importing PromptTemplate for prompts
import markdown
from html2docx import html2docx
import os
import openai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
#import langchain
#langchain.debug = True
def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
         sadrzaj = infile.read()
         infile.close()
         return sadrzaj
# Creating a list of messages that includes a system message and an AI message
def main():

    st.title('Large Text Summarizer with Input for .pdf, .txt and .docx')  # Setting the title for Streamlit application
    uploaded_file = st.file_uploader("Choose a file")

    openai.api_key = os.environ.get('OPENAI_API_KEY')  # Reading OpenAI API key from file
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key)  # Initializing ChatOpenAI model

    placeholder = st.empty()
    st.session_state['question'] = ''
    # document.add_heading('Suma velikog dokumenta', level=1)
    dld="blank"
    buf = html2docx("nothing", title="Summary")
    # summarize chosen file
    if uploaded_file is not None:
    
        with placeholder.form(key='my_form', clear_on_submit=True): 
               # st.write(uploaded_file.name)
               with open(uploaded_file.name, "wb") as file:
                        file.write(uploaded_file.getbuffer())
               if ".pdf" in uploaded_file.name:
                    loader = UnstructuredPDFLoader(uploaded_file.name, encoding="utf-8")
               else:
                    loader = UnstructuredFileLoader(uploaded_file.name, encoding="utf-8")  # Creating a file loader object

               result = loader.load()  # Loading text from the file

               text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)  # Creating a text splitter object
               texts = text_splitter.split_documents(result)  # Splitting the loaded text into smaller chunks
               prompt_initial=open_file("prompt_initial.txt")
               prompt_final=open_file("prompt_final.txt")
               prompt_opsirni= open_file("prompt_opsirni.txt")
               opis1 = st.text_input(f"Ovo je postojeci inicijalni prompt :  {prompt_initial}  Dodajte potrebne detalje koji ce zauzeti mesto polja opis1 detalje koji nedostaju: ")
               opis2 = st.text_input(f"Ovo je postojeci finalni prompt : {prompt_final} Dodajte potrebne detalje koji ce zauzeti mesto polja opis2 detalje koji nedostaju: ")
               st.write(f"Ovo je postojeci prompt za opsirni deo teksta : {prompt_opsirni} ")
               submit_button = st.form_submit_button(label='Submit')
               # Creating a list of messages that includes a system message and an AI message
               opp = PromptTemplate(template=prompt_opsirni, input_variables=["text"]) 
               initial= PromptTemplate(template=prompt_initial, input_variables=["text" , "opis1"])  # Creating a prompt template object
               final = PromptTemplate(template=prompt_final, input_variables=["text", "opis2" ])  # Creating a prompt template object
              

               
           
               if submit_button:
                    with st.spinner("Sacekajte trenutak..."):
                        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False, map_prompt=initial, combine_prompt=final)
                        # Load the summarization chain with verbose mode
                        chain2 = load_summarize_chain(llm, chain_type="map_reduce", verbose=False, return_intermediate_steps=True, map_prompt=opp, combine_prompt=opp)
                        prosireno = chain2({"input_documents": texts}, return_only_outputs=True)
                        samo_text = prosireno['intermediate_steps']

                        output_string = ""
                        # Process each element of the list
                        for i, step in enumerate(samo_text, start=1):
                            # Create a variable dynamically for each element
                            var_name = f"Poglavlje {i}"
                            globals()[var_name] = step
                            output_string += f" **{var_name}:**  {step}\n\n"
                        
                        st.markdown("# Opsirnije" + "\n\n")
                        st.markdown(output_string)
                        st.markdown("\n\n" + "# Ukratko" + "\n\n")
                        suma = AIMessage(content=chain.run(input_documents=texts, opis1=opis1, opis2=opis2))
                        st.markdown(suma.content)  # Displaying the summary
                        dld =  "# Executive Summary" + "\n\n" +suma.content + "\n\n" +  "## Opsirnije" + "\n\n" + output_string 
                        html = markdown.markdown(dld)
                        buf = html2docx(html, title="Summary")
                        
        st.download_button(
                label="Click here to download",
                data=buf.getvalue(),
                file_name="Suma.docx",
                mime="docx"
                )   

st.set_page_config(
page_title="Positive summarizer",
page_icon="📖",
layout="wide",
initial_sidebar_state="collapsed",
)



hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login to use the service', 'main')

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main', key='unique_key')
    # if login success run the program
    main()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
 
   

