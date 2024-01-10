import streamlit as st
import pandas as pd
import openai
import base64
import os

from functions.show_data_info import show_data_info
from functions.improve_prompt import improve_prompt
from functions.run_prompts_app import run_prompts_app

from src.keboola_storage_api.connection import add_keboola_table_selection
from src.st_aggrid.st_aggrid import interactive_table

image_path = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Kai PromptLab", page_icon=image_path+"/static/keboola.png", 
                   layout="wide"
                   )

logo_image = image_path+"/static/keboola_logo.png"
logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(logo_image, "rb").read()).decode()}" style="width: 150px; margin-left: -10px;"></div>'
st.markdown(f"{logo_html}", unsafe_allow_html=True)

st.title('Kai PromptLab 👩🏻‍🔬')

OPENAI_API_KEY = st.sidebar.text_input('Enter your OpenAI API Key:', value="Enter API Key",
    help= """
    You can get your own OpenAI API key by following these instructions:
    1. Go to https://platform.openai.com/account/api-keys.
    2. Click on the __+ Create new secret key__ button.
    3. Enter an identifier name (optional) and click on the __Create secret key__ button.
    """,
    type="password", 
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_uploaded_file(upload_option):
    if upload_option == 'Connect to Keboola Storage':
        add_keboola_table_selection()
        st.session_state.setdefault('uploaded_file', None)
    elif upload_option == 'Use Demo Dataset':
        file = image_path + "/data/sample_data.csv"
        st.session_state['uploaded_file'] = file
    return st.session_state['uploaded_file']

def display_main_content(uploaded_file, openai_api_key):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        show_data_info(df)
        
        if st.session_state['uploaded_file'] is not None:
            interactive_table()
        
        if len(openai_api_key) < 14:
            st.warning("To continue, please enter your OpenAI API Key.")
        
        improve_prompt()
        run_prompts_app(df)

        st.text(" ")
        st.markdown(f"{logo_html}", unsafe_allow_html=True)
    else:
        st.markdown("""
        __Welcome to the PromptLab!__ 
        
        🔄 __Connect__
                    
        - Start by connecting to the Keboola storage, you'll need your API token to do this. To get it, go to _Settings_ in your Keboola account and find the _API Tokens_ tab (see the [documentation](https://help.keboola.com/management/project/tokens/) for more information).
        You will then be able to select the bucket and table you want to work with. 
                    
        - You'll also need an OpenAI API key; if you don't have one, you can get it [here](https://platform.openai.com/account/api-keys).

        __Once connected, you can use the app in the following steps:__
                    
        📊 __1. Explore__ – In this section, you'll see the uploaded table to make sure you are working with the correct data.
                    
        🛠️ __2. Improve__ – If you have ideas but are unsure about the wording of your prompts, have them improved here. Simply enter your idea and hit the _Improve_ button. The app will return an improved version that follows prompt engineering best practices.
                   
        🤹‍♂️ __3. Test__ – This is where you can experiment and fine-tune your prompts. You can input 1-3 prompts to run with your data. Each prompt comes with its own settings, allowing you to tweak parameters or compare results across different models.
         
        If you ever feel lost, you can always find this information in the _Guide_ tab.
                    """)

def main():
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    upload_option = st.sidebar.selectbox('Select an upload option:', 
                                    ['Connect to Keboola Storage',
                                    #'Upload a CSV file',
                                    'Use Demo Dataset'
                                     ], help="""
    You can get your own API token by following these instructions:
    1. Go to Settings in your Keboola account.
    2. Go to the __API Tokens__ tab.
    3. Click on __+ NEW TOKEN__ button, set it and __CREATE__.
    """)

    uploaded_file = get_uploaded_file(upload_option)

    tab1, tab2 = st.tabs(["App", "Guide"])
    with tab1:
        display_main_content(uploaded_file, OPENAI_API_KEY)
    
    with tab2:
        st.markdown("""
                    
        🔄 __Connect__ – Start by connecting to the Keboola storage, you'll need your API token to do this. Go to _Settings_ and find the _API Tokens_ tab. Once connected, you'll be able to select the bucket and table you want to work with. 
        
        📊 __Explore__ – Here you'll see the uploaded table to make sure you are working with the correct data.
                    
        🛠️ __Improve__ – If you have ideas but are unsure about the wording of your prompts, use this section. Simply enter your idea and hit the _Improve_ button. The app will return an improved version of your prompt that follows prompt engineering best practices.
                   
        🤹‍♂️ __Test__ – This is where you can experiment and fine-tune your prompts. You can input 1-3 prompts to run with your data. Each prompt comes with its own settings, allowing you to tweak parameters or compare results across different models. Additionally, you can specify the portion of your dataset you want to work with.
        
        💌 __Feedback__ – If you have any questions, encounter issues, or have suggestions for improving the PromptLab, don't hesitate to reach out! andrea.novakova@keboola.com
        
        🔗 __Useful Links__
        - Keboola's [API Tokens](https://help.keboola.com/management/project/tokens/)
        - Get your OpenAI API key [here](https://platform.openai.com/account/api-keys)
        - Read the prompt engineering [best practices](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
        - PromptLab [GitHub repo](https://github.com/keboola/kai-promptlab/tree/main)

                    """)
    
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()