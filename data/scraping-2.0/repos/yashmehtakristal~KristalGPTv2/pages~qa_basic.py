import streamlit as st
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages

# Setting page config & header
st.set_page_config(page_title="Kristal Retriever", page_icon="📖", layout="wide")
st.header("📖 Kristal Retriever")

import openai
from streamlit_extras.app_logo import add_logo
from st_pages import Page, Section, add_page_title, show_pages, hide_pages



## Importing functions
from bundle import no_embeddings_process_documents_individual, embeddings_process_documents_individual
from core.chroma import st_server_file, upload_zip_files


### CODE

add_logo("https://assets-global.website-files.com/614a9edd8139f5def3897a73/61960dbb839ce5fefe853138_Kristal%20Logotype%20Primary.svg")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY

# Error handling for OpenAI API key
if not openai_api_key:
    st.warning(
        "There is something wrong with the API Key Configuration."
        "Please check with creator of the program (OpenAI keys can be found at https://platform.openai.com/account/api-keys)"
    )

if 'username' in st.session_state:
    st.session_state.username = st.session_state.username

def change_states():
    st.session_state.logged_out = True
    st.session_state.logged_in = False
    st.session_state.password_match = None

# Display app only if user is logged in
if st.session_state.logged_in is True and st.session_state.logout is False:

    st.sidebar.subheader(f'Welcome {st.session_state.username}')

    #st.session_state.Authenticator.logout('Log Out', 'sidebar')
    # logout_button = st.session_state.Authenticator.logout('Log Out', 'sidebar')
    logout_button = st.sidebar.button("Logout", on_click = change_states)

    # Check embeddings
    check_embeddings = st.radio(label = "Do you have saved embeddings?", options = ["Yes", "No"], index = None, help = "Embeddings are saved files created by ChromaDB", disabled=False, horizontal = False, label_visibility="visible")


    # User does not have embeddings they can use
    if check_embeddings == "No":

        master_folder, chroma_file_path, chroma_file_name = st_server_file()

        uploaded_files = st.file_uploader(
        "Upload your pdf documents",
        type=["pdf"],
        help="You can upload multiple files."
        "Please note that scanned documents are not supported yet!",
        accept_multiple_files = True)


    # User has embeddings which they can use
    elif check_embeddings == "Yes":

        uploaded_zip_file = upload_zip_files()

        uploaded_files = st.file_uploader(
        "Upload your pdf documents",
        type=["pdf"],
        help="You can upload multiple files."
        "Please note that scanned documents are not supported yet!",
        accept_multiple_files = True
    )

    else:
        st.warning("Please select whether you have embeddings to use or not")
        st.stop()

    # Display the question input box for user to type question and submit
    with st.form(key="qa_form"):

        query = st.text_area(label = "Ask a question from the documents uploaded", value = None, height = None, max_chars = None, help = "Please input your questions regarding the document. Greater the prompt engineering, better the output", disabled = False, label_visibility = "visible")
        submit = st.form_submit_button("Submit")

        if not query:
            st.warning("Please enter a question to ask about the document!")
            st.stop()

    if submit:
        if check_embeddings == "No":
            if uploaded_files:
                no_embeddings_process_documents_individual(uploaded_files = uploaded_files, chroma_file_path = chroma_file_path, prompt = query)
            else:
                st.warning(
                    "1) Please upload the pdf files",
                    icon="⚠")
                st.stop()

        elif check_embeddings == "Yes":
            if uploaded_files:
                embeddings_process_documents_individual(uploaded_files = uploaded_files, prompt = query, uploaded_zip_file = uploaded_zip_file)

            else:
                st.warning(
                    "1) Please upload the excel files",
                    icon="⚠")
                st.stop()

else:
    st.info("Seems like you are not logged in. Please head over to the Login page to login", icon="ℹ️")
