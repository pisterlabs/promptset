import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import tempfile

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# Function to save the uploaded file to a temporary file
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error('Error saving uploaded file.')
        return None

# Function to load and index the CSV data
@st.cache_resource
def load_and_index_data(file_path):
  
    # Load the documents
    loader = CSVLoader(file_path=file_path)
    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    # Create a question-answering chain using the index
    # chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-4-1106-preview"), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
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
    if saved_file_path:
        st.session_state.chain = load_and_index_data(saved_file_path)

# Chat UI
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with the alumni database?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling form submission
if st.session_state['chain']:
    with st.form(key='question_form'):
        user_input = st.text_input("Ask a question about the alumni:", key="user_input")
        submit_button = st.form_submit_button("Submit")

        if submit_button and user_input:
            # Get the response from the chatbot
            response = st.session_state.chain({"question": user_input})
            
            # Use 'result' to access the answer
            if 'result' in response:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response['result']})
            else:
                # Handle the case where 'result' is not in the response
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I could not find an answer."})