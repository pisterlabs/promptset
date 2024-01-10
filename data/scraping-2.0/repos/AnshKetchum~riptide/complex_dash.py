import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader

from leadgen.agents.base import get_chain 

from leadgen.utils.doc_utils import extract_text_from_upload
from leadgen.utils.latex_utils import generate_latex, template_commands, render_latex
from leadgen.prompts.resume import generate_json_resume
from leadgen.llms.current import provider


import json

import os


# Chat UI title
st.header("JobsGPT")
st.subheader('File type supported: PDF/DOCX/TXT :city_sunrise:')

openai_api_key = os.getenv("OPENAI_API_KEY")
agent = None

embeddings = provider.get_embeddings()

#if os.path.exists("data/user_docs.faiss"):
#    vectorstore = FAISS.load_local(os.path.join("data", "user_docs."), embeddings)
#else:
vectorstore = FAISS.from_texts(["This is some starting text"], embeddings)

# File uploader in the sidebar on the left
with st.sidebar:
    if not openai_api_key:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Set OPENAI_API_KEY as an environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

        
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    st.info("Please refresh the browser if you decided to upload more files to reset the session", icon="🚨")
    
# Check if files are uploaded
if uploaded_files:
    # Print the number of files to console
    print(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded PDF files
        documents = []
        for uploaded_file in uploaded_files:
            # Get the full file path of the uploaded file
            file_path = os.path.join(os.getcwd(), uploaded_file.name)

            # Save the uploaded file to disk
            with open(os.path.join("data", "uploaded", file_path), "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = UnstructuredFileLoader(file_path)
            loaded_documents = loader.load()
            print(f"Number of files loaded: {len(loaded_documents)}")

            # Extend the main documents list with the loaded documents
            documents.extend(loaded_documents)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)

        vectorstore.add_documents(document_chunks)

        print('Saving locally')
        vectorstore.save_local("data", index_name="user_docs")

        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

        # Print the number of total chunks to console
        print(f"Number of total chunks: {len(document_chunks)}")

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your questions?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the assistant using the latest chat history
    print("MESSAGES")
    messages = [(message["role"], message["content"]) for message in st.session_state.messages]
    print(messages)

    if not agent:
        agent = get_chain(vectorstore, st)

    result = agent({"input": prompt, "chat_history": messages})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = result["output"]
        message_placeholder.markdown(full_response + "|")
    message_placeholder.markdown(full_response)    
    print(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
