
from code_reader.reader_config import *

import os
import uuid
import subprocess
import glob
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import streamlit as st

import sys
sys.path.append(r'/Users/eeilstein/Desktop/Code/Python/Repos/RepoReader')  # Provide the absolute path to project_root

from general_utils import (
    extract_repo_name,
    is_repo_cloned,
    clone_github_repo,
    clone_repo,
    is_directory_empty,
    get_openai_api_key,
    )
from general_config import (github_url, stat_path_repos as STAT_PATH_REPOS)

### load_and_index_files


def load_and_index_files(repo_path):
    if is_directory_empty(repo_path):
        st.error("ERROR: The repo is not cloned. Please clone the repo first.")
        return None
    #     extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala',
    #                   'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css',
    #                   'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']
    extensions = ['py', 'sql', 'yml', 'md']
    file_type_counts = {}
    documents_dict = {}

    texts = []
    for ext in extensions:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4_000, chunk_overlap=200)
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext == 'ipynb':
                # Get a list of all .ipynb files in the directory
                notebook_files = glob.glob(f'{repo_path}/**/*.{ext}', recursive=True)

                loaded_documents = []
                # Load each file using NotebookLoader
                for file in notebook_files:
                    if 'archive' in file.split(os.sep):
                        continue  # Skip files in 'archive' folder
                    loader = NotebookLoader(file, include_outputs=True, max_output_length=20, remove_newline=True)
                    documents = loader.load()
                    loaded_documents += documents

            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern, loader_kwargs={"content_type": "text/plain"})
                loaded_documents = []
                if callable(loader.load):
                    all_documents = loader.load()
                    # Exclude documents located in 'archive' folders
                    for doc in all_documents:
                        file_path = doc.metadata['source']
                        if 'archive' not in file_path.split(os.sep):
                            loaded_documents.append(doc)
            #                         if ext == 'py':
            #                             print(file_path)

            if loaded_documents:
                print(f'[LOG] {ext} loaded!')
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id

                    documents_dict[file_id] = doc

            texts += text_splitter.split_documents(loaded_documents)
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            # print(traceback.format_exc())
            continue

    return texts




### Process answer func

def process_llm_response(llm_response):
    answer = llm_response['result']
    print(GREEN + '\nANSWER\n\n' + answer + RESET_COLOR + '\n')
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    return llm_response['result']



### [Optional] Create it from scratch

def create_vectordb(local_path, repo_name, embedding, persist_directory):
    print('Creating VectorDB from scratch')
    st.warning('Creating VectorDB from scratch...')
    ### [Optional] Index the repo files and create the VectorDB if it doesn't exist

    ### Process the repo from local dir
    texts = load_and_index_files(os.path.join(local_path, repo_name))

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    print('VectorDB Created')
    st.success('VectorDB Created')

    # persiste the db to disk
    vectordb.persist()
    vectordb = None
    print('Persisting VectorDB on local')


    return vectordb


### pull from repo
def pull_repo(github_url, local_path):
    repo_name = extract_repo_name(github_url)
    repo_path = os.path.join(local_path, repo_name)
    print(f'[LOG] Pulling repo {repo_name}...')
    try:
        subprocess.run(['git', 'pull'], cwd=repo_path, check=True)
        st.success(f'Repo {repo_name} pulled successfully!')
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull repository: {e}")
        st.error(f'Failed to pull repository: {e}')

### Reset history

def reset_history():
    st.session_state['conversation_history'] = ""


### Process answer func


## Call the LLM API

def get_llm_api(model_name, temperature):
    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
    )


# Start chatting!

def chat_with_llm_model(query, qa_chain, repo_name, github_url, context_template):

    print('Thinking...')

    kw = {"repo_name": repo_name, "github_url": github_url, "conversation_history": st.session_state['conversation_history'], }
    st.session_state.update(kw)
    llm_response = qa_chain(context_template.format(question=query, **kw))
    # result, sources = process_llm_response(llm_response)
    result, sources = llm_response['result'], llm_response["source_documents"]

    # remove duplicates in sources
    sources = list(set([source.metadata['source'] for source in sources]))

    # st.write("conversation_history\n" + kw['conversation_history'])

    return result, sources


# show sources in sidebar
def show_sources(sources):
    """
    show sources in sidebar
    :param sources:list
    :return: None
    """
    for source in sources:
        st.sidebar.markdown(format_source(source), unsafe_allow_html=True)


def format_question(text):
    return f'<span style="color:purple">{text}</span>'


def format_answer(text):
    return f'<span style="color:green">{text}</span>'


def format_exception(text):
    return f'<span style="color:red">{text}</span>'


def format_source(text):
    return f'<span style="color:grey">{text}</span>'


def print_color(text, color):
    print(f"{color}{text}{RESET_COLOR}\n")