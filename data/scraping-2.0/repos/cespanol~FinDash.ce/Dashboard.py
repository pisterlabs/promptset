import os
import datetime
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from werkzeug.utils import secure_filename
import streamlit as st
from streamlit.components.v1 import html







# import plaid
# from plaid_link_component.plaid_link import plaid_link

# PLAID_CLIENT_ID = st.secrets['plaid_client_id']
# PLAID_SECRET = st.secrets['plaid_secret']
# PLAID_PUBLIC_KEY = st.secrets['plaid_public_key']
# PLAID_ENV = 'sandbox'  # Change to 'development' or 'production' when necessary

# client = Client(client_id=PLAID_CLIENT_ID, secret=PLAID_SECRET, public_key=PLAID_PUBLIC_KEY, environment=PLAID_ENV)

# def handle_public_token(public_token):
#     exchange_response = client.Item.public_token.exchange(public_token)
#     st.session_state.plaid_access_token = exchange_response['access_token']
#     st.success("Connected to Plaid.")


#session_states - Initialize session state values if not already set
if "api_key" not in st.session_state:
    st.session_state.api_key = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = False





def get_openai_api_key():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = False
    if not st.session_state.api_key:
        # Create an empty container for the input box and the warning message
        input_container = st.empty()
        warning_container = st.empty()

        openai_api_key = input_container.text_input("Enter your OpenAI API Key:")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = st.secrets['openai_api_key']

            # Remove the input box and the warning message by emptying the containers
            input_container.empty()
            warning_container.empty()
            # input_container.write('~ OpenAI API Key in use')
            # st.success('OpenAI - GPT3.5 Turbo')

            st.success('OpenAI API key in use')
            st.session_state.api_key = True
            return True
        else:
            warning_container.warning("Please enter your OpenAI API Key.")
            return False
    

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        # Create an empty container for username and password inputs
        login_label_container = st.empty()
        login_label_container.subheader('Login')
        login_container = st.empty()
        with login_container:
            with st.form(key='my_form'):
                username = st.text_input('Username:')
                password = st.text_input('Password:', type='password')
                login_submitted = st.form_submit_button('Login')
            if login_submitted:
                if username == st.secrets['username'] and password == st.secrets['password']:
                    login_container.empty()
                    login_label_container.empty()
                    st.session_state.logged_in = True
                    st.session_state.username = username
                else:
                    st.error('Incorrect username or password')





def main():
    
    #Page Config
    st.set_page_config(page_title="FinDash.ce", layout="wide")
    
    #Page Title
    welcome_container = st.empty()
    welcome_container.title('FinDash.ce')

    #API Key Input
    get_openai_api_key()

    #Sidebar during login
    sidebar = st.sidebar
    with sidebar:
        sidebar_title_container = st.empty()
        sidebar_title_container.title('FinDash.ce')
        sidebar_label_1_container = st.empty()
        sidebar_settings_container = st.empty()
        sidebar_settings_container.header('Settings')

        #API Key Status
        if st.session_state.api_key == False:
            st.warning('Please enter your OpenAI API key')
        if st.session_state.api_key == True:
            st.success('OpenAI API key in use')

        #Login Status
        if st.session_state.logged_in == False:
            sidebar_label_1_container.write('Please log in to continue')
       
    
    #Login after API key
    if st.session_state.api_key == True:
        login_label_container = st.empty()
        login_label_container.write('Please log in to continue')
        login()



    #Logged In - User Main Interface
    if st.session_state.logged_in:
        welcome_container.empty()
        login_label_container.empty()
        sidebar_label_1_container.empty()
        sidebar_settings_container.success('Logged in as {}'.format(st.session_state.username))
# 
      
        #Logged in - Sidebar Financials Menu 
        pages = sidebar_title_container.selectbox("Menu", ('Accounts','Investments', 'Transactions','Dashboard', 'Categories', 'Recurrings', 'Savings' ),3)
        st.session_state.pages = pages
  
        
        #Logged in - Mainbar
        st.header('Dashboard')

        
        if st.session_state.pages == 'Accounts':
            pass
            # plaid_link(handle_public_token)

        
        if st.session_state.pages == 'Investments':
            st.subheader('Total Balance')
            st.write('Graph of total balance over time')
            st.subheader('Investment Accounts')
            st.subheader('Allocation')
            st.write('This is the text')
        
        
        if st.session_state.pages == 'Transactions':
            st.subheader('This is the subheader')
            st.write('This is the text')
        
        
        if st.session_state.pages == 'Dashboard':
            st.subheader('This is the subheader')
            st.write('This is the text')
        
        
        if st.session_state.pages == 'Categories':
            st.subheader('This is the subheader')
            st.write('This is the text')
        
        
        if st.session_state.pages == 'Recurrings':
            st.subheader('This is the subheader')
            st.write('This is the text')
        
        
        if st.session_state.pages == 'Savings':
            st.subheader('This is the subheader')
            st.write('This is the text')





if __name__ == "__main__":
    main()