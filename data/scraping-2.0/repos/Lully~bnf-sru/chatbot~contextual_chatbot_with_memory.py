# coding: utf-8

# source : https://github.com/azur-scd/streamlit-gpt-experimentations/blob/main/pages/Contextual_document_chatbot_with_memory.py

# Import necessary libraries
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import OnlinePDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set Streamlit page configuration
st.set_page_config(page_title="Bot on loaded document", layout="wide")

# Set up the Streamlit app layout
st.title("Contextual ChatBot with memory on custom document")

# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


# Define function to get user input
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    return st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Ask anything ...",
        label_visibility="hidden",
    )


# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    del st.session_state.stored_session
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    # st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()
    index = None


# Initialize params
MODEL = "gpt-3.5-turbo"
index = None

# Set up the Streamlit app layout
col1, col2, col3 = st.columns(3)
with col1:
    option = st.selectbox("How would you want to do ?", ("Load PDF", "Read online PDF"))
with col2:
    API_O = st.text_input(
        ":blue[Put Your OPENAI API-KEY :]",
        placeholder="Paste your OpenAI API key here ",
        type="password",
    )
with col3:
    K = st.slider(
        " Number of prompts to display in te conversation",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

if option == "Load PDF":
    if uploaded_file := st.file_uploader("**Upload Your PDF File**", type=["pdf"]):
        pdf_reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=4000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(openai_api_key=API_O)
        with st.spinner("Loading and indexing..."):
            index = FAISS.from_texts(texts, embeddings)
            # index.save_local("faiss_index") puis index = FAISS.load_local("faiss_index", embeddings)
        st.success("Done.", icon="✅")
elif option == "Read online PDF":
    if file_url := st.text_input("**Paste an url**"):
        loader = OnlinePDFLoader(file_url).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        docs = text_splitter.split_documents(loader)
        print(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=API_O)
        with st.spinner("Loading and indexing..."):
            index = FAISS.from_documents(docs, embeddings)
        st.success("Done.", icon="✅")
else:
    st.warning("A file must be loaded to try the ChatBot on it")

if (API_O != "") & (index is not None):
    llm = ChatOpenAI(
        temperature=0, openai_api_key=API_O, model_name=MODEL, verbose=False
    )
    # Create a ConversationEntityMemory object if not already created
    st.session_state.entity_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    Conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(),
        memory=st.session_state.entity_memory,
    )
else:
    st.warning("You need to set your OpenAI API-KEY and provide a document to index")

if user_input := get_text():
    output = Conversation.run({"question": user_input})
    print(output)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


# Add a button to start a new chat
st.button("New Chat", on_click=new_chat, type="primary")

# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])
