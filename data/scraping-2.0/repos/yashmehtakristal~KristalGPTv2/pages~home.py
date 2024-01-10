import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title = "Kristal Retriever", page_icon = "📖", layout = "wide", initial_sidebar_state = "expanded")
st.header("📖 Kristal Retriever")

# Add the logo to the sidebar
add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")

import openai
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY

if 'username' in st.session_state:
    st.session_state.username = st.session_state.username

def change_states():
    st.session_state.logged_out = True
    st.session_state.logged_in = False
    st.session_state.password_match = None

# let User see app if logged in = True & logged out = False
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    # Display Markdown of the main page
    st.markdown(
    '''
    This section will give more information about Kristal GPT. 

    This application has 2 main features (Bulk Upload and Q&A). Moreover, it has two high-level categorization (Basic, Advanced) 

    Here is the simple categorization of the aforementioned:

    - Basic
        - Bulk Upload - Basic
        - Q&A - Basic
    - Advanced
        - Bulk Upload - Advanced
        - Q&A - Advanced

    ### Features explanation

    ***Bulk Upload:***

    This feature allows the user to upload an excel file (or select a template) containing the list of prompts, along with other relevant fields.

    ***Q&A:***

    This feature allows the user to input prompts individually, as if they are "chatting" with the uploaded documents.

    ### Categorization

    ***Basic:***

    The Basic version of the application has the minimum features required to successfully run the application. These are:

    1. Option to save embeddings for current iteration/load saved embeddings
    2. Specifying the folder for the embeddings
    3. Uploading the pdf files, as well as the excel files.
    4. Displaying the results as a dataframe
    5. Providing option to download displayed dataframe as a CSV file or Excel file

    ***Advanced:***

    The Advanced version of the application has the same features as the basic, with the addition of the following:

    1. Select which LLM model to use
    2. Select the number of nodes to retrieve from LLM (during vector search)
    3. Select the temperature parameter of LLM
    4. Select the request timeout (in seconds) of LLM
    5. Select the maximum retries of LLM
    6. Select the amount of time for LLM to wait before executing next prompt (in loop)
    7. Select whether to display all chunks retrieved from vector search (If no, i.e. default value, will display the chunk that has highest score)
    8. Select to show the parsed contents of the document
    9. Select to show all tables parsed from the pdf document
    '''
    )

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
