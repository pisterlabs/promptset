import os
import streamlit as st
import shutil
import time
import subprocess
import torch


from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings


def list_files_with_size(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_info = [{"Filename": file, "Size (bytes)": os.path.getsize(os.path.join(folder_path, file))} for file in files]
    return file_info

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            break
        size_bytes /= 1024.0
    return "{:.2f} {}".format(size_bytes, unit)

def copy_files_to_folder(upload_files, folder_path):
    for upload_file in upload_files:
        target_path = os.path.join(folder_path, upload_file.name)
        with open(target_path, "wb") as f:
            f.write(upload_file.getbuffer())

# Set a 1:1 layout ratio
st.set_page_config(layout="wide")

# Add a title
st.title("Files for Embeddings")

# Organize the layout with the file upload section on the left and the table on the right
col1, col2 = st.columns(2)

# Initial data
folder_path = "web/docs_to_db"  # Update with your folder path

# File uploader for adding new files on the left
with col1:
    if st.button("Open File Explorer"):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        docs_folder = os.path.join(current_dir, "docs_to_db")

        # Ensure the directory exists
        if not os.path.exists(docs_folder):
            os.mkdir(docs_folder)

        # Open the directory in Windows File Explorer
        subprocess.Popen(f'explorer "{docs_folder}"')

    uploaded_files = st.file_uploader("Copy files to folder", type=("pdf"), accept_multiple_files=True)
    # Trigger the action directly after submitting the files
    if uploaded_files is not None:
        # Handle file upload
        copy_files_to_folder(uploaded_files, folder_path)

# Display the list of files in the table with formatted size on the right
with col2:
    # Use a placeholder for the table
    table_placeholder = st.empty()

    # Function to update the table content
    def update_table():
        files_info = list_files_with_size(folder_path)
        table_content = {"Filename": [info["Filename"] for info in files_info],
                         "Size": [format_size(info["Size (bytes)"]) for info in files_info]}
        table_placeholder.table(table_content)

    # Update the table initially
    update_table()


    # Add a button to manually refresh the list of files
    if st.button("Refresh List", use_container_width=True):
        # Simulate refreshing data
        time.sleep(2)  # Simulate a delay for refreshing data
        # Update the table content
        update_table()

# Add a line underneath
st.markdown("***")

col31,col32 = st.columns([1,1])

with col31:
    st.title("Transformer Model Settings")

    # Get the list of transformer models from the "models/transformer" folder
    transformer_folder_path = "models/transformer"
    transformer_options = [f for f in os.listdir(transformer_folder_path) if os.path.isdir(os.path.join(transformer_folder_path, f))]

    # Add a dropdown menu to select a transformer
    selected_transformer = st.selectbox("Choose a Transformer:", transformer_options)

    with st.expander("Show Files..."):
        # Get the list of files for the selected transformer
        selected_transformer_path = os.path.join(transformer_folder_path, selected_transformer)
        files_in_transformer = [f for f in os.listdir(selected_transformer_path) if os.path.isfile(os.path.join(selected_transformer_path, f))]

        # Display the list of files
        st.write("Files in selected transformer:")
        st.write(files_in_transformer)

    col321, col322 = st.columns([1,1])
    with col321:
            
        # 1. Dropdown menu for "Device"
        device_options = ["CPU", "CUDA"]
        selected_device = st.selectbox("Device", device_options, index=0)  # Default index set to 0 (CPU)
        # selected_device = st.radio("Device", device_options, index=0)  # Default index set to 0 (CPU)


        # 3. Dropdown field for "Context"
        context_options = list(range(1, 11))
        default_context = 4
        selected_context = st.selectbox("Num Context", context_options, index=context_options.index(default_context))

    with col322:

        # 4. Number input for "Chunk size"
        default_chunk_size = 512
        selected_chunk_size = st.number_input("Chunk size", value=default_chunk_size)

        # 5. Number input for "Chunk overlap"
        default_chunk_overlap = 64
        selected_chunk_overlap = st.number_input("Chunk overlap", value=default_chunk_overlap)

    # 2. Value changer for "Similarity"
    default_similarity = 0.9
    selected_similarity = st.slider("Similarity", 0.0, 1.0, default_similarity, 0.1)


with col32:
    st.title("Create Vector DB")

    with st.expander("Settings Oveview:"):
        st.write(f"Selected Transformer: {selected_transformer}")
        st.write(f"Selected Device: {selected_device}")
        st.write(f"Selected Similarity: {selected_similarity}")
        st.write(f"Selected Context: {selected_context}")
        st.write(f"Selected Chunk size: {selected_chunk_size}")
        st.write(f"Selected Chunk overlap: {selected_chunk_overlap}")

        with st.echo():
            ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
            SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/docs_to_db"
            PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/Vector_DB"
            INGEST_THREADS = os.cpu_count() or 8

            CHROMA_SETTINGS = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=PERSIST_DIRECTORY,
                anonymized_telemetry=False
            )

            EMBEDDING_MODEL_NAME = selected_transformer

# if st.button("Create Vector DB", use_container_width=True):
        