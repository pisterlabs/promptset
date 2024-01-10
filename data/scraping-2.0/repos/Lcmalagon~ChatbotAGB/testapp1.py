import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import time
import os
import tempfile

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# Function to save the uploaded file to a temporary file
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            saved_path = tmp_file.name
            print(f"File saved successfully at {saved_path}")
            return saved_path
    except Exception as e:
        st.error('Error saving uploaded file.')
        print(f"Error saving file: {e}")
        return None

# Function to load and index the CSV data
@st.cache_resource
def load_and_index_data(file_path):
    start_time = time.time()

    print("Loading CSV...")
    loader = CSVLoader(file_path=file_path)
    load_time = time.time()
    print(f"CSV Loaded in {load_time - start_time} seconds")

    print("Creating Index...")
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    index_time = time.time()
    print(f"Index Created in {index_time - load_time} seconds")

    print("Creating Chain with GPT-4-1106-preview model...")
    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    
    chain_time = time.time()
    print(f"Chain Created in {chain_time - index_time} seconds")

    total_time = time.time()
    print(f"Total load_and_index_data time: {total_time - start_time} seconds")
    return chain

# Initialize the chatbot chain in the session state
if "chain" not in st.session_state:
    st.session_state['chain'] = None

# Streamlit UI for uploading the CSV and initializing the chatbot
st.title("Alumni Database Chatbot")

file_upload = st.sidebar.file_uploader("Please Upload the Alumni CSV", type=['csv'])

# If a file is uploaded, save it and load/index the data
if file_upload is not None:
    saved_file_path = save_uploaded_file(file_upload)
    print(f"Uploaded file saved at {saved_file_path}")
    if "last_file_path" not in st.session_state or st.session_state.last_file_path != saved_file_path:
        print("Initializing new chain with uploaded file...")
        st.session_state.chain = load_and_index_data(saved_file_path)
        st.session_state.last_file_path = saved_file_path
        print("Chain initialized successfully.")
    else:
        print("Using existing chain.")

# Chat UI
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with the alumni database?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling form submission
if st.session_state.get('chain'):
    with st.form(key='question_form'):
        user_input = st.text_input("Ask a question about the alumni:", key="user_input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        print(f"User input: {user_input}")
        # Get the response from the chatbot
        response = st.session_state.chain({"question": user_input})
        print(f"Chatbot response: {response}")

        # Use 'result' to access the answer
        if 'result' in response:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
        else:
            # Handle the case where 'result' is not in the response
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I could not find an answer."})
