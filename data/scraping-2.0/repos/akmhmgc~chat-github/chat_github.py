import pickle
import os
import re
import uuid
import streamlit as st
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, LLMPredictor, download_loader
download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader

st.title("OpenAI and GitHub API App")
st.write("Enter your OpenAI API key and GitHub token to start.")

openai_api_key = st.text_input("OpenAI API Key", value="", type="password")
github_token = st.text_input("GitHub Token", value="", type="password")

index = None
if 'download_file_name' not in st.session_state:
    st.session_state['download_file_name'] = None

if 'docs' not in st.session_state:
    st.session_state['docs'] = None

if 'index' not in st.session_state:
    st.session_state['index'] = None

if openai_api_key and github_token:
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key))
    st.write("API keys have been set.")
    
    # Add input field for repository URL
    repo_url = st.text_input("Repository URL (required)", value="")
    branch_name = st.text_input("Branch name (required)", value="")
    filter_directories = st.text_input("Filter Directories (optional, comma-separated)", value="")
    filter_file_extensions = st.text_input("Filter File Extensions (optional, comma-separated)", value="")

    # Process the optional fields
    filter_directories = tuple(filter_directories.split(',')) if filter_directories else None
    filter_file_extensions = tuple(filter_file_extensions.split(',')) if filter_file_extensions else None

    # Parse owner and repo from the URL
    url_pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?github\.com\/([^\/]+)\/([^\/]+)")
    match = url_pattern.match(repo_url)
    owner, repo = match.groups() if match else (None, None)

    if st.button("Make index data") and owner and repo and branch_name and not index:
        # Show a loading message while the data is being created
        with st.spinner("Loading data..."):
            download_file_name = uuid.uuid4().hex
            st.session_state.download_file_name = download_file_name
            # Load the data
            github_client = GithubClient(github_token)
            loader = GithubRepositoryReader(
                github_client,
                owner=owner,
                repo=repo,
                filter_directories=(filter_directories, GithubRepositoryReader.FilterType.INCLUDE) if filter_directories else None,
                filter_file_extensions=(filter_file_extensions, GithubRepositoryReader.FilterType.INCLUDE) if filter_file_extensions else None,
                verbose=True,
                concurrent_requests=10,
            )
            docs = loader.load_data(branch=branch_name)
            st.session_state.docs = docs
            with open(f'download_{download_file_name}.pkl', 'wb') as f:
                pickle.dump(docs, f)
            st.success("Data loaded successfully!")

    is_file = os.path.isfile(f'download_{st.session_state.download_file_name}.pkl')
    if is_file:
        with open(f'download_{st.session_state.download_file_name}.pkl', "rb") as f:
            tmp = f.read()
            st.download_button(
                label="Download index data",
                data=tmp,
                file_name="docs.pkl",
                mime="application/octet-stream",
            )
        with open(f'download_{st.session_state.download_file_name}.pkl', "rb") as f:
            st.session_state.docs = pickle.load(f)

    uploaded_file = st.file_uploader("Upload index file (Optional)")

    if uploaded_file and st.button("Upload index data"):
        upload_file_name = uuid.uuid4().hex
        with open(f'upload_{upload_file_name}.pkl', "wb") as f:
            f.write(uploaded_file.getvalue())
        with open(f'upload_{upload_file_name}.pkl', "rb") as f:
            st.session_state.docs = pickle.load(f)
    
    if st.session_state.docs and not st.session_state.index:
        st.session_state.index = GPTSimpleVectorIndex(st.session_state.docs, llm_predictor=llm_predictor)

    if st.session_state.index:
        index = st.session_state.index
        # Show the question input field after the data is loaded
        user_question = st.text_input("Enter your question:", value="")
        if user_question:
            output = index.query(user_question)
            st.write("Response:")
            st.markdown(f"<h3 style='font-size: 18px;'>{output}</h3>", unsafe_allow_html=True)

            st.write("Source:")
            st.markdown(f"<h3 style='font-size: 18px;'>{output.source_nodes[0].extra_info['file_path']}</h3>", unsafe_allow_html=True)
