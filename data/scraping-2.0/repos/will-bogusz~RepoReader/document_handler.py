import os
import tiktoken
import subprocess
import shutil
import stat
import uuid
import streamlit as st
import chromadb
from langchain.document_loaders.generic import GenericLoader
from langchain.schema.document import Document
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from chromadb.utils import embedding_functions
import openai
import time
from utils import get_working_collection, call_with_timeout, get_openai_key
from chat_handler import get_chunk_classification, call_with_timeout_translation, call_with_timeout_embed
import chardet
import PyPDF2
import requests
import json
import threading
import queue

COST_PER_TOKEN = 0.0001 / 1000  # $0.0001 per 1K tokens
MODEL_NAME = 'gpt-3.5-turbo'

def get_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        return chardet.detect(file.read())['encoding']

def set_permissions_recursive(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IWRITE)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IWRITE)

def is_text_file(file_path):
    try:
        encoding = get_file_encoding(file_path) or 'utf-8'
        with open(file_path, 'r', encoding=encoding) as file:
            file.read()
        return True
    except UnicodeDecodeError:
        return False
    
def remove_with_sudo(path):
    try:
        subprocess.run(['sudo', 'rm', '-rf', path], check=True)
        print(f"Successfully removed {path}")
    except subprocess.CalledProcessError as e:
        print(f"Error removing {path}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def clean_cloned_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.normpath(os.path.join(root, file))
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension != '.pdf' and not is_text_file(file_path):
                print(f"Removing file: {file_path}")
                os.remove(file_path)

def clone_repo(git_url):
    local_path = "/home/will/hosting/RepoReader/Documents/Repositories"
    repo_name = git_url.strip('/').rstrip('.git').split('/')[-1]
    full_local_path = os.path.normpath(os.path.join(local_path, repo_name))

    if os.path.isdir(full_local_path):
        print(f"Repository already exists locally at '{full_local_path}'")
        return full_local_path

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    try:
        subprocess.run(['git', 'clone', git_url, full_local_path], check=True)
        print(f'Repository cloned successfully to {full_local_path}')

        set_permissions_recursive(full_local_path)

        # Remove .git, .gitignore, and .github files/folders after cloning
        git_dir = os.path.normpath(os.path.join(full_local_path, '.git'))
        gitignore_file = os.path.normpath(os.path.join(full_local_path, '.gitignore'))
        github_dir = os.path.normpath(os.path.join(full_local_path, '.github'))

        if os.path.exists(git_dir):
            remove_with_sudo(git_dir)
        if os.path.exists(github_dir):
            remove_with_sudo(github_dir)
        if os.path.exists(gitignore_file):
            remove_with_sudo(gitignore_file)

        time.sleep(1)
        clean_cloned_directory(full_local_path)

        return full_local_path
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while cloning the repository: {e}')
        return ""
    except Exception as e:
        print(f'Something went wrong: {e}')
        return ""

def prompt_for_urls():
    # Define the key for our dynamic inputs in the session state
    if 'url_inputs' not in st.session_state:
        st.session_state.url_inputs = ['']

    # Define function to add a new URL input
    def add_url_input():
        st.session_state.url_inputs.append('')

    # Define function to remove the last URL input
    def remove_url_input():
        if len(st.session_state.url_inputs) > 1:
            st.session_state.url_inputs.pop()

    # Display the URL inputs dynamically
    for i, _ in enumerate(st.session_state.url_inputs):
        st.session_state.url_inputs[i] = st.text_input(f"URL {i+1}", value=st.session_state.url_inputs[i], key=f"url_{i}")

    # Add and remove URL buttons
    col1, col2 = st.columns(2)
    with col1:
        add_button = st.button("Add URL", on_click=add_url_input)
    with col2:
        remove_button = st.button("Remove Last URL", on_click=remove_url_input)

    # Button to submit URLs
    submit = st.button('Submit')
    if submit:
        st.write("Submitted URLs:")
        repo_final_paths = []
        for url in st.session_state.url_inputs:
            if url:  # Check if the URL is not empty
                st.write(url)
                # Call clone_repo and collect the cloned repo paths
                repo_path = clone_repo(url)
                if repo_path:
                    repo_final_paths.append(repo_path)
                # Provide feedback on successful clone
                st.success(f"Cloned {url} successfully!")
            else:
                # Provide feedback for empty URL field
                st.error("Please enter a URL.")

def embed_text(text, key):
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002",
        "encoding_format": "float"
    }

    passed = False
    for j in range(5):
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                passed = True
                break
            elif response.status_code == 429:  # Rate limit error
                time.sleep(2 ** j)
            else:
                break  # Break on other errors
        except Exception as e:
            print(f"Exception occurred: {e}")
            break

    if not passed:
        raise RuntimeError("Failed to create embeddings.")
    
    embedding = response.json()['data'][0]['embedding']
    return embedding

def get_language_from_extension(file_path):
    _, ext = os.path.splitext(file_path)
    language_mapping = {
        '.py': ('Python', 'Source Code'),
        '.js': ('JavaScript', 'Source Code'),
        '.java': ('Java', 'Source Code'),
        '.c': ('C', 'Source Code'),
        '.cpp': ('C++', 'Source Code'),
        '.cs': ('C#', 'Source Code'),
        '.ts': ('TypeScript', 'Source Code'),
        '.php': ('PHP', 'Source Code'),
        '.rb': ('Ruby', 'Source Code'),
        '.go': ('Go', 'Source Code'),
        '.swift': ('Swift', 'Source Code'),
        '.kt': ('Kotlin', 'Source Code'),
        '.rs': ('Rust', 'Source Code'),
        '.lua': ('Lua', 'Source Code'),
        '.groovy': ('Groovy', 'Source Code'),
        '.r': ('R', 'Source Code'),
        '.sh': ('Shell', 'Source Code'),
        '.bat': ('Batch', 'Source Code'),
        '.ps1': ('PowerShell', 'Source Code'),
        '.pl': ('Perl', 'Source Code'),
        '.scala': ('Scala', 'Source Code'),
        '.h': ('C/C++ Header', 'Source Code'),
        '.hpp': ('C++ Header', 'Source Code'),
        '.html': ('HTML', 'Source Code'),
        '.css': ('CSS', 'Source Code'),
        '.xml': ('XML', 'Source Code'),
        '.json': ('JSON', 'Data/Text'),
        '.yaml': ('YAML', 'Configuration'),
        '.yml': ('YAML', 'Configuration'),
        '.md': ('Markdown', 'Data/Text'),
        '.csv': ('CSV', 'Data/Text'),
        '.txt': ('Text', 'Data/Text'),
        '.sql': ('SQL', 'Source Code'),
        '.dart': ('Dart', 'Source Code'),
        '.f': ('Fortran', 'Source Code'),
        '.vb': ('Visual Basic', 'Source Code'),
        '.jsx': ('JSX', 'Source Code'),
        '.tsx': ('TSX', 'Source Code'),
        '.ini': ('INI', 'Configuration'),
        '.toml': ('TOML', 'Configuration'),
    }

    return language_mapping.get(ext.lower(), ('Text', 'Data/Text'))


def clean_path(path, base_path):
    return path.replace(base_path, '', 1).lstrip('\\/')

def get_source_from_path(path):
    parts = path.split(os.sep)
    if 'Repositories' in parts:
        return parts[parts.index('Repositories') + 1]  # Repository name
    elif 'Uploaded' in parts:
        return 'Uploaded'
    return 'Unknown'

def process_document(doc_path, base_path, results_queue):
    full_path = os.path.normpath(doc_path)
    if not os.path.exists(full_path):
        print(f"Error: File not found {full_path}")
        return

    file_name = os.path.basename(full_path)
    cleaned_path = clean_path(full_path, base_path)
    language, file_type = get_language_from_extension(full_path)
    source = get_source_from_path(full_path)

    metadata = {
        'filename': file_name,
        'filepath': cleaned_path,
        'language': language,
        'source': source,
        'type': file_type,
        'translation': None
    }

    file_extension = os.path.splitext(full_path)[1].lower()
    if file_extension == '.pdf':
        try:
            with open(full_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                pdf_text = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_text.append(page.extract_text())
                content = ' '.join(pdf_text)
        except Exception as e:
            print(f"Error reading PDF file {full_path}: {e}")
            return
    else:
        encoding = get_file_encoding(full_path) or 'utf-8'
        try:
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(content)

    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=process_chunk, args=(chunk, metadata, file_type, results_queue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def process_chunk(chunk, metadata, file_type, results_queue):
    doc_metadata = metadata.copy()
    if file_type == 'Source Code':
        translation, error = call_with_timeout_translation(get_chunk_classification, [chunk, metadata])
        if error:
            print(f"Error or timeout on first try translating. Retrying...")
            translation, error = call_with_timeout_translation(get_chunk_classification, [chunk, metadata])
            if error:
                print(f"Error or timeout on second try translating. Using default behavior.")
                print(f"Failed to translate a document for: {metadata['filename']}")
                translation = None
        
        if translation:
            doc_metadata['translation'] = chunk
            chunk = translation

    doc = Document(page_content=chunk, metadata=doc_metadata)
    results_queue.put(doc)

def batch_add_documents_to_collection(documents, collection, key):
    # Prepare lists to store batch data
    batch_documents = []
    batch_embeddings = []
    batch_metadatas = []
    batch_ids = []

    for document in documents:
        doc_id = str(uuid.uuid4())

        # Generate embedding for each document
        embedding, error = call_with_timeout_embed(embed_text, [document.page_content, key])
        if error:
            print(f"Error or timeout on first try embedding {error}. Retrying...")
            embedding, error = call_with_timeout_embed(embed_text, [document.page_content, key])
            if error:
                print(f"Error or timeout on second try embedding. Retrying...")
                embedding, error = call_with_timeout_embed(embed_text, [document.page_content, key])
                if error:
                    print(f"Error or timeout on third try embedding. Skipping...")
                    print(f"Unable to add document, failed embedding: {doc_id} | {document.metadata['filename']}")
                    continue  # Skip this document if embedding fails

        # Process metadata
        document.metadata = stringify_dictionary(document.metadata)

        # Append data to respective batches
        batch_documents.append(document.page_content)
        batch_embeddings.append(embedding)
        batch_metadatas.append(document.metadata)
        batch_ids.append(doc_id)

    # Add the batch of documents to the collection
    if batch_documents:
        collection.add(
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

        # Optionally, print each added document
        for doc_id, metadata in zip(batch_ids, batch_metadatas):
            print(f"Added Document: {doc_id} | {metadata['filename']}")

def batch_process_documents(results_queue, collection, key, batch_size=10):
    batch_documents = []

    while True:
        try:
            document = results_queue.get(timeout=5)  # Adjust timeout as needed
            if document is None:  # Sentinel value to indicate completion
                # Process any remaining documents in the batch
                if batch_documents:
                    batch_add_documents_to_collection(batch_documents, collection, key)
                break

            batch_documents.append(document)
            if len(batch_documents) >= batch_size:
                batch_add_documents_to_collection(batch_documents, collection, key)
                batch_documents = []

        except queue.Empty:
            # If no items are in the queue, process any remaining documents in the batch
            if batch_documents:
                batch_add_documents_to_collection(batch_documents, collection, key)
                batch_documents = []

def store_documents(docs):
    base_path = "/home/will/hosting/RepoReader/Documents/"

    if not docs:
        # Traverse the directories and subdirectories to get all file paths
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                docs.append(file_path)

    collection = get_working_collection()
    results_queue = queue.Queue()
    threads = []

    key = get_openai_key()

     # Start batch processing thread
    batch_thread = threading.Thread(target=batch_process_documents, args=(results_queue, collection, key))
    batch_thread.start()

    for doc_path in docs:
        thread = threading.Thread(target=process_document, args=(doc_path, base_path, results_queue))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Signal the batch processing thread to finish
    results_queue.put(None)
    batch_thread.join()

    total_embeddings = collection.count()
    print(f"Collection now has {total_embeddings} embeddings!")

# helper to clean up metadata
def stringify_dictionary(input_dict):
    return {str(key): (str(value) if not isinstance(value, str) else value) for key, value in input_dict.items()}

def upload_documents():
    valid_data_documents = ["doc", "txt", "md", "pdf", "log", "py", "js"]
    base_path = "/home/will/hosting/RepoReader/Documents/Uploaded"

    document = st.file_uploader("Upload your data", type=valid_data_documents)
    if document is not None:
        file_path = os.path.join(base_path, document.name)

        with open(file_path, "wb") as f:
            f.write(document.getbuffer())
        st.success("File successfully uploaded!")

def count_tokens(text):
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return len(encoding.encode(text))

def calculate_cost(text):
    token_count = count_tokens(text)
    return token_count * COST_PER_TOKEN

def total_cost_for_documents(documents):
    return sum(calculate_cost(doc) for doc in documents)

def calculate_cost_from_selection(selected_files, base_path):
    documents_content = []
    file_count = 0

    if selected_files:
        for file_path in selected_files:
            file_extension = os.path.splitext(file_path)[1].lower()
            try:
                if file_extension == '.pdf':
                    # Read from PDF file
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        pdf_text = []
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            pdf_text.append(page.extract_text())
                        documents_content.append(' '.join(pdf_text))
                else:
                    # Read from other file types using detected encoding
                    encoding = get_file_encoding(file_path) or 'utf-8'
                    with open(file_path, 'r', encoding=encoding) as f:
                        documents_content.append(f.read())
                file_count += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    else:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                try:
                    if file_extension == '.pdf':
                        # Read from PDF file
                        with open(file_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            pdf_text = []
                            for page_num in range(len(pdf_reader.pages)):
                                page = pdf_reader.pages[page_num]
                                pdf_text.append(page.extract_text())
                            documents_content.append(' '.join(pdf_text))
                    else:
                        # Read from other file types using detected encoding
                        encoding = get_file_encoding(file_path) or 'utf-8'
                        with open(file_path, 'r', encoding=encoding) as f:
                            documents_content.append(f.read())
                    file_count += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    if documents_content:
        total_cost = total_cost_for_documents(documents_content)
        return total_cost, file_count
    else:
        return None, 0