import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from io import BytesIO
import zipfile
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    set_global_service_context,
    download_loader
)
from llama_index.llms import OpenAI

def load_and_unzip_file(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall('/tmp')
    repo_name = uploaded_file.name.replace(".zip", "")
    clone_path = f"/tmp/{repo_name}"
    GPTRepoReader = download_loader("GPTRepoReader")
    loader = GPTRepoReader()
    documents = loader.load_data(repo_path=clone_path)
    return documents

def delete_unzipped_folder(uploaded_file):
    repo_name = uploaded_file.name.replace(".zip", "")
    clone_path = f"/tmp/{repo_name}"
    if os.path.exists(clone_path):
        shutil.rmtree(clone_path)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Git Repository",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.sidebar.subheader("Open AI API Key, Find it here: https://platform.openai.com/api-keys")
    api_key = st.sidebar.text_input("Enter your API key", type="password")

    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    st.header("Chat with Git Repository :books:")

    with st.sidebar:
        st.subheader("Your Repository")
        uploaded_file = st.file_uploader("Upload your repository as a .zip file and click on 'Process'", type=['zip'])
        if st.button("Process"):
            with st.spinner("Processing"):
                # load and unzip file
                documents = load_and_unzip_file(uploaded_file)

                # Set up the service context with the desired model
                service_context = ServiceContext.from_defaults(
                    llm=OpenAI(model="gpt-3.5-turbo-16k", temperature=0, max_tokens=14384, api_key=api_key)
                )
                set_global_service_context(service_context)

                # create an index from the documents
                index = VectorStoreIndex.from_documents(documents)

                # Use the index as a chat engine
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="context")

                # delete unzipped folder
                delete_unzipped_folder(uploaded_file)

    user_question = st.text_input("Ask a question about your repository:")
    if user_question and 'chat_engine' in st.session_state:
        # Construct a contextual prompt
        base_context = "Here is a collection of files from a GitHub repository. "
        contextual_prompt = base_context + user_question

        # Use the chat engine to ask the contextual question
        response = st.session_state.chat_engine.chat(contextual_prompt)
        st.write(response)


if __name__ == '__main__':
    main()
