### Imports

import os
import uuid
import subprocess
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import DirectoryLoader, NotebookLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from utils import clean_and_tokenize
import unicodedata
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from openai.error import InvalidRequestError
from chromadb.errors import NoIndexException, NotEnoughElementsException
# from SessionState import get

import streamlit as st

# Get the session state for this session.
st.session_state['conversation'] = []

### Config

# colors
# ======#

WHITE = "\033[37m"
GREEN = "\033[32m"
PURPLE = "\033[35m"
RED = "\033[31m"
RESET_COLOR = "\033[0m"
GREY = "\033[90m"

# LLM vars
# =========#

LLM_TEMPERATURE = 0.2
MODEL_NAME = "gpt-3.5-turbo-16k"  # VERIFIED
# model_name = "gpt-3.5-turbo" # VERIFIED
# model_name = "gpt-4-32k" # NOT YET
# model_name = "gpt-4" # NOT YET


# Prompt vars
# ============#

if 'conversation_history' not in st.session_state.keys():
    st.session_state['conversation_history'] = ""

if 'last_q' not in st.session_state.keys():
    st.session_state['last_q'] = ""

context = """Repo: {repo_name} ({github_url}) | | Conversation history: {conversation_history}

            Instructions:
            1. Answer based on context/docs.
            2. Focus on repo/code.
            3. Consider:
                a. Purpose/features - describe.
                b. Functions/code - provide details/samples.
                c. Setup/usage - give instructions.
            4. SQL Syntax is Bigquery.
            5. Unsure? Say "I am not sure".


    Question: {question}
    Answer:
"""

local_path = 'stat_path_repos'


### load_and_index_files


def load_and_index_files(repo_path):
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


#### repo funcs

def extract_repo_name(repo_url):
    # Extract the part of the URL after the last slash and before .git
    repo_name = repo_url.rstrip().rstrip('/').split('/')[-1]
    if len(repo_name) < 2:
        err_msg = f"Invalid repo URL: {repo_url}"
        st.error(err_msg)
        raise ValueError(err_msg)
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]  # remove .git from the end
    return repo_name


def is_repo_cloned(repo_url, path_dir):
    repo_name = extract_repo_name(repo_url)
    repo_path = os.path.join(path_dir, repo_name)
    return os.path.isdir(repo_path)


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False


### Process answer func

def process_llm_response(llm_response):
    answer = llm_response['result']
    print(GREEN + '\nANSWER\n\n' + answer + RESET_COLOR + '\n')
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    return llm_response['result']


### Clone repo

def clone_repo(github_url):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    repo_name = extract_repo_name(github_url)
    local_path = 'stat_path_repos'

    _is_repo_cloned = is_repo_cloned(github_url, local_path)
    print(f'[LOG] is repo {repo_name} already cloned? {_is_repo_cloned}')

    # if the repo is already cloned in the static path, then skip cloning. If not, clone it in the static path
    if _is_repo_cloned:
        st.success(f'Repo {repo_name} already cloned')
    else:
        st.warning(f'Cloning repo {repo_name}...')
        clone_github_repo(github_url, os.path.join(local_path, repo_name))
        st.success(f'Repo {repo_name} is now cloned!')


    return repo_name, _is_repo_cloned


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

def chat_with_llm_model(query, qa_chain, repo_name, github_url):
    global context

    print('Thinking...')

    kw = {"repo_name": repo_name, "github_url": github_url, "conversation_history": st.session_state['conversation_history'], }
    st.session_state.update(kw)
    llm_response = qa_chain(context.format(question=query, **kw))
    # result, sources = process_llm_response(llm_response)
    result, sources = llm_response['result'], llm_response["source_documents"]

    # remove duplicates in sources
    sources = list(set([source.metadata['source'] for source in sources]))

    # st.write("conversation_history\n" + kw['conversation_history'])

    return result, sources


def format_question(text):
    return f'<span style="color:purple">{text}</span>'


def format_answer(text):
    return f'<span style="color:green">{text}</span>'


def format_exception(text):
    return f'<span style="color:red">{text}</span>'


def format_source(text):
    return f'<span style="color:grey">{text}</span>'


### Main


def main(repo_url, num_src_docs, is_reset_history, HARD_RESET_DB=False):
    global LLM_TEMPERATURE, LLM_MODEL_NAME, show_history

    if is_reset_history: reset_history()

    ### Clone repo from Github
    repo_name, is_repo_cloned = clone_repo(repo_url)


    slider = st.slider(
        label='Num of Relevant Docs Input', min_value=1,
        max_value=30, value=num_src_docs, key='docs_slider')

    num_src_docs = slider

    persist_directory = f'chroma_db_{repo_name}'
    embedding = OpenAIEmbeddings()

    ### Reset our Chroma Vector DB
    try:
        # # To cleanup, you can delete the collection
        vectordb.delete_collection()
        vectordb.persist()
    except:
        print("No vectordb found. Proceeding...")

    if (not os.path.exists(persist_directory)) or (not (is_repo_cloned)) or HARD_RESET_DB:
        create_vectordb(local_path, repo_name, embedding, persist_directory)

    # Now we can load the persisted database from disk, and use it as normal
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    print('Called existing VectorDB')

    ### Create a retriever from the indexed DB
    retriever = vectordb.as_retriever(search_kwargs={"k": num_src_docs})

    st.write("Ask a question about the repository, BE SPECIFIC ('exit' to quit)")
    ## Check the retriever
    try:
        docs = retriever.get_relevant_documents("What is the name of the repo?")
        print(len(docs))
    except Exception as e:
        if isinstance(e, NoIndexException):
            create_vectordb(local_path, repo_name, embedding, persist_directory)
        elif isinstance(e, NotEnoughElementsException):
            err_msg = f"=== Try reducing the 'Relevant Docs' slider (currently {num_src_docs}) ===\n"
            st.error(e.__str__())
            st.write(
                format_exception(err_msg),
                unsafe_allow_html=True)

    ## Call the LLM API
    turbo_llm = get_llm_api(MODEL_NAME, LLM_TEMPERATURE)

    ### Integrate LLM API and source-docs in the Chain

    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           verbose=True, )

    ### START CHATTING!


    conversation = []
    # This will hold the current query. Whenever the user submits a new query, it gets added here.
    query = st.text_input("Your question:")

    # is query changed?
    is_query_changed = st.session_state['last_q'] != query


    # Check if the user has entered a new query.
    if query and is_query_changed:
        try:
            st.session_state['last_q'] = query
            st.session_state['conversation_history'] += f'Last Question: {query} \n'

            print(f'\n{PURPLE}QUESTION\n\n{query}{RESET_COLOR}\n')
            # Perform the chat operation.
            result, sources = chat_with_llm_model(query, qa_chain, repo_name, repo_url)

            st.session_state['conversation_history'] += f'Last Answer: {result}\n\n'
            print(f"{GREEN}ANSWER\n\n{result}{RESET_COLOR}\n")

            for src in sources:
                print(f"{GREY}SOURCE\n\n{src}{RESET_COLOR}\n")

            # Add this interaction to the conversation history.
            st.session_state.conversation.append((query, result, sources))

            # Display the question, answer, and sources.
            st.write(format_question(f"Question: {query}"), unsafe_allow_html=True)
            st.write(format_answer(f"Answer: {result}"), unsafe_allow_html=True)
            # st.subheader("Sources:")
            # for source in sources:
            #     st.write(format_source(source.metadata['source']), unsafe_allow_html=True)

            for source in sources:
                st.sidebar.markdown(format_source(source), unsafe_allow_html=True)


        except Exception as e:
            st.write(format_exception(e), unsafe_allow_html=True)
            if isinstance(e, InvalidRequestError) or isinstance(e, NotEnoughElementsException):
                err_msg = f"=== Try reducing the 'Relevant Docs' slider (currently {num_src_docs}) ===\n"
                st.write(
                    format_exception(err_msg),
                    unsafe_allow_html=True)

    # Show history
    if show_history:
        st.subheader("Conversation History")
        st.write(st.session_state['conversation_history'])

    # Display the conversation history in the sidebar.
    # formatted_conversation = "\n".join([
    #     format_question(f"You: {user_query}") + "<br>" + format_answer(
    #         f"Answer: {chat_answer}") + "<br>" + format_source(
    #         f"Source: {[source.metadata['source'] for source in sources]}")
    #     for user_query, chat_answer, sources in st.session_state.conversation
    # ])
    #
    # st.sidebar.markdown(formatted_conversation, unsafe_allow_html=True)

    print('len conversation:', len(conversation))

    #
    # try:
    # except:
    #     pass

    # st.sidebar.title("Chat History")
    # for user_query, chat_answer, sources in conversation:
    #     st.sidebar.write(format_question(f"You: {user_query}"), unsafe_allow_html=True)
    #     st.sidebar.write(format_answer(f"Answer: {chat_answer}"), unsafe_allow_html=True)
    #     for source in sources:
    #         st.sidebar.write(format_source(source.metadata['source']),  unsafe_allow_html=True)

    # break


# User Input
# ========== #

# GitHub URL
github_url = r"https://github.com/Lightricks/dwh-data-model-transforms"

# Num Relevant Docs
NUM_SOURCE_DOCS = 20

# want to reset history?
is_reset_history = False

# Reset Chroma DB? // Usually False
HARD_RESET_DB = False


# ========== #



st.title("Interactive Chat with LLM API")
st.sidebar.title("Sources")

if __name__ == "__main__":
    input_url = st.text_input("GitHub URL", github_url)
    github_url = input_url

    start_col, reset_col, history_col = st.columns([1,1,2])

    start_btn = start_col.checkbox("Start Chatting")
    HARD_RESET_DB = reset_col.checkbox("Reset Chroma DB?", value=False)
    show_history = history_col.checkbox("Show History", value=False)

    if start_btn:
        main(github_url, NUM_SOURCE_DOCS, is_reset_history, HARD_RESET_DB)
