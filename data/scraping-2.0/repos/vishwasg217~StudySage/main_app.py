from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

import streamlit as st

from utils import process_pdf, database

model = None


def handle_query(query: str):
    result = st.session_state.conversation({"question": query, "chat_history": ""})
    history = st.session_state.memory.load_memory_variables({})['chat_history']
    print(st.session_state.memory.load_memory_variables({})['chat_history'])
    for i, msg in enumerate(history):
        if i%2 == 0:
            st.write("hello")
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)

if __name__ == "__main__":

    if "memory" not in st.session_state:
        st.session_state.memory = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.set_page_config(layout="wide", page_title="StudySage", page_icon="📚")
    st.title(":books: StudySage - Your AI Study Buddy")
    st.divider()

    
    st.session_state.OPEN_AI_API = st.sidebar.text_input("Open AI API Key", type="password")

    if st.session_state.OPEN_AI_API:    
        model = ChatOpenAI(openai_api_key=st.session_state.OPEN_AI_API, model_name="gpt-3.5-turbo")

        if "process_pdf" not in st.session_state:
            st.session_state.process_pdf = False

        pdfs = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)
        if st.sidebar.button("Process PDF"):
            st.session_state.process_pdf = True
            with st.spinner("Processing PDF..."):
                splitted_text = process_pdf(pdfs)
                db = database(splitted_text)
            # system_message =  """You are a study assistant chatbot where a student can upload their notes and you answer question based on it.
            #               explain the answer in simple manner and neatly formatted way. Provide examples wherever needed. Don't assume the student to have too much prior knowledge.
            #               You must stick to the information provided in the notes/document. 
            #               """
            st.session_state.memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, k=5)
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(llm=model, retriever=db.as_retriever(), memory=st.session_state.memory)
            # st.session_state.memory.save_context({"input": system_message}, {"output": "Sure I will do exactly as per the requirements"})

        if st.session_state.process_pdf:
            query = st.chat_input("Ask a question")
            if query:
                handle_query(query)

    
    

    

