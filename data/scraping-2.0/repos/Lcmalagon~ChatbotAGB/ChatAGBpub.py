import streamlit as st
<<<<<<< HEAD
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import tempfile

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = "your chatgpt api here"

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
    # Load the documents from CSV
    loader = CSVLoader(file_path=file_path)
    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
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
=======
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import os

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = "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"

uploaded_file = st.sidebar.file_uploader("Upload", type="csv")

# Streamlit UI for uploading the CSV and initializing the chatbot
st.title("Chat AGB")

st.markdown("---")

# Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>I'm your Alumni Database Chatbot, powered by LangChain and Streamlit. 
    I leverage advanced language models to provide insightful answers from your alumni database. 
    Upload a CSV file and ask me questions about the alumni data.</h5>
    """,
    unsafe_allow_html=True)
st.markdown("---")

if uploaded_file:
    print("Uploaded file detected.")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        print(f"Temporary file path: {tmp_file_path}")

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    print(f"Number of documents loaded: {len(data)}")
else:
    data = None
    print("No file uploaded.")

if data is not None:
    embeddings = OpenAIEmbeddings()
    print("Embeddings created.")
    vectorstore = FAISS.from_documents(data, embeddings)
    print("Vectorstore created.")
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever()
    )
    print("Conversational retrieval chain created.")
else:
    vectorstore = None
    chain = None
    print("Data is None, skipping embeddings, vectorstore, and chain creation.")

def conversational_chat(query):
    print(f"Received query: {query}")
    if chain is not None:
        result = chain({"question": query, "chat_history": st.session_state['history']})
        print(f"Result from chain: {result}")
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    else:
        print("Chain is None, returning default message.")
        return "Sorry, no data is available."

if 'history' not in st.session_state:
    st.session_state['history'] = []

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Ask questions about your alumni database here", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        print(f"User input submitted: {user_input}")
        output = conversational_chat(user_input)
        print(f"Output generated: {output}")
        # Append both user input and chatbot response to chat history
        st.session_state['history'].append(("Question", user_input))
        st.session_state['history'].append(("Answer", output))

# Display the chat history in the Streamlit UI
with response_container:
    for role, message in st.session_state['history']:
        if role == "Question":
            st.markdown(f"**Question:** {message}")
        elif role == "Answer":
            st.markdown(f"**Answer:** {message}")
>>>>>>> 4079ef4 (Initial commit)
