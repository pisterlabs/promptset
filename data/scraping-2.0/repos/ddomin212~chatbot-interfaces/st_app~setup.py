import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers, Replicate
from langchain.memory import ConversationBufferMemory
from streaming_handler import StreamingResponseHandler
from langchain.vectorstores import FAISS
from io import BytesIO

def init_st_session_state() -> None:
    """Initializes the session state variables for chatting. These include the chat history, the generated responses, and the past user inputs."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_st_elements() -> tuple[float, list[BytesIO]]:
    """Renderes various streamlit elements on the page."""
    
    # setup page
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("RAG Chatbot ❓ 🦜")

    # setup sidebar
    st.sidebar.title("Upload your company documents")
    uploaded_files = st.sidebar.file_uploader("Supported formats include: .pdf, .docx, .doc, .txt, .csv", accept_multiple_files=True)
    temp_slider = st.sidebar.slider("Temperature", min_value=0.01, max_value=1.0, value=0.15, step=0.05, key="Temperature")
    
    return temp_slider, uploaded_files

def create_conversational_chain(vector_store: FAISS, temp = 0.15) -> ConversationalRetrievalChain:
    """Creates a chatbot chain with memory

    Args:
        vector_store: vector db (FAISS)

    Returns:
        chatbot chain
    """

    # define model
    llm = CTransformers(model="../llama-2-7b-chat.ggmlv3.q4_0.bin",
                        streaming=True, 
                        callbacks=[StreamingResponseHandler()],
                        model_type="llama", config={"max_new_tokens": 512, "temperature": temp})

    # define memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # create chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain