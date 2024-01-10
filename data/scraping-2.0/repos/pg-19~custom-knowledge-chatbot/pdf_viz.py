from langchain.document_loaders import PyPDFLoader
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
import streamlit as st
import tempfile

st.title("Custom Knowledge Chatbot")

apikey = st.sidebar.text_input("Put your OpenAI API key here")
os.environ["OPENAI_API_KEY"] = apikey
pdf_file = st.sidebar.file_uploader("Upload your PDF file")
csv_file = st.sidebar.file_uploader("Upload your CSV file")

if not (pdf_file or csv_file) or not apikey:
    st.write("You can start a session with the chatbot after inputting your API key and uploading your PDF/CSV file!")
else:
    # Loads the PDF/CSV file and split it into pages
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        if pdf_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
        elif csv_file:
            tmp_file.write(csv_file.getvalue())
            tmp_file_path = tmp_file.name

            loader = CSVLoader(file_path=tmp_file_path)
            pages = loader.load_and_split()

    # Loads the text into the FAISS index/vector store
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

    # Stores the chat history and tracks context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), faiss_index.as_retriever(), memory=memory)

    # Chatbot function that takes in the user's query and returns the chatbot's response
    def conversational_chat(query):
                
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
                
        return result["answer"]

    # Formats the chat window       
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'chatbot' not in st.session_state:
        st.session_state['chatbot']= ["Hello! Feel free to ask me any questions about your PDF"]

    if 'user' not in st.session_state:
        st.session_state['user'] = ["Hi!"]
            
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    # Chat window that takes in the user's input and displays the chat history
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask your questions here", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['user'].append(user_input)
            st.session_state['chatbot'].append(output)

    if st.session_state['chatbot']:
            with response_container:
                for i in range(len(st.session_state['chatbot'])):
                    message(st.session_state["user"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["chatbot"][i], key=str(i), avatar_style="thumbs")


   

