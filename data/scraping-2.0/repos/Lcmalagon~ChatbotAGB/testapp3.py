import streamlit as st
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

if uploaded_file:
    print("Uploaded file detected.")
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        print(f"Temporary file path: {tmp_file_path}")

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    print(f"Number of documents loaded: {len(data)}")
else:
    # Handle the case when no file is uploaded
    data = None
    print("No file uploaded.")

if data is not None:
    embeddings = OpenAIEmbeddings()
    print("Embeddings created.")
    vectorstore = FAISS.from_documents(data, embeddings)
    print("Vectorstore created.")
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-4-1106-preview'),
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
    print("Initialized 'history' in session state.")

if 'generated' not in st.session_state:
    if uploaded_file is not None:
        file_name = uploaded_file.name
        print(f"Uploaded file name: {file_name}")
    else:
        file_name = "the data"  # Default text when no file is uploaded
        print("No file uploaded, using default file name.")
    st.session_state['generated'] = ["Hello! Ask me anything about " + file_name + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

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
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display the chat history
with response_container:
    for past_input, generated_output in zip(st.session_state['past'], st.session_state['generated']):
        st.text(past_input)
        st.text(generated_output)
