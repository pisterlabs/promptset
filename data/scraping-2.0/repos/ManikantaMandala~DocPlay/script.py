import pickle
from docx import Document
import streamlit as st
import streamlit_authenticator as stauth
from dependencies import sign_up, fetch_users
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css,bot_template,user_template
from dotenv import load_dotenv
load_dotenv()

#This will generate text from the pdfs, docs uploaded and return the text
def get_text(docs):
    text = ""
    for doc in docs:
        file_name = doc.name
        text+= "The file name is %s\n"%file_name
        if(file_name.endswith('.pdf')):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        if(file_name.endswith('.docs')):
            docs_reader = Document(doc)
            for paragraph in docs_reader.paragraphs:
                text += paragraph.text + "\n"
    return text

#This will take the text, create the chunks and return those chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

#This will take GooglePalmEmbeddings, and makes a local vector_store using FAISS
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

#This will read the previous text which is in the server
def read_previous_text(username):
    data_file = f"data_{username}.obj"

    try:
        with open(data_file, 'rb') as readFile:
            previous_txt = pickle.load(readFile)
        return previous_txt
    except FileNotFoundError:
        return ""

#This will write the updated text into the data files in the server
def write_text_data_file(username,previous_text):
    data_file = f"data_{username}.obj"
    with open(data_file, 'wb') as writeFile:
        pickle.dump(previous_text, writeFile)
    writeFile.close()

def get_conversational_chain(vector_store):
    llm=GooglePalm()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def submit():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    chat_history_length = len(st.session_state.chatHistory)

    # Iterate over chat history in reverse order
    for i in range(chat_history_length - 2, -1, -2):
        user_message = st.session_state.chatHistory[i]
        bot_message = st.session_state.chatHistory[i + 1]
        st.write(user_template.replace("{{MSG}}", user_message.content), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", bot_message.content), unsafe_allow_html=True)

def main():
    try:
        st.set_page_config("DocPlay 💬")
        st.header("DocPlay 💬")
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []
    
        for user in users:
            emails.append(user['key'])
            usernames.append(user['username'])
            passwords.append(user['password'])
        
        credentials = {'usernames': {}}
        for index in range(len(emails)):
            credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}
        
        Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)
        email, authentication_status, username = Authenticator.login(':green[Login]', 'main')

        info, info1 = st.columns(2)

        ##check sign up
        if not authentication_status:
            sign_up()

        if username:
            if username in usernames:
                if authentication_status:
                    #let user see the app
                    st.write(css, unsafe_allow_html=True)

                    # st.header("Chat with Multiple PDF 💬")
                    st.text_input("Ask a Question from the PDF Files", key="widget", on_change=submit)
                    if "user_question" not in st.session_state:
                        st.session_state.user_question = ""
                    user_question = st.session_state.user_question
                    if "conversation" not in st.session_state:
                        st.session_state.conversation = None
                    if "chatHistory" not in st.session_state:
                        st.session_state.chatHistory = None
                    if "clear_history_pressed" not in st.session_state:
                        st.session_state.clear_history_pressed = False
                    if user_question:
                        user_input(user_question)
                    with st.sidebar:
                        st.title(f'Welcome {username}')
                        st.subheader("Upload your Documents")
                        checkbox_container = st.empty()
                        previous_data = checkbox_container.checkbox("Previous text", disabled = st.session_state.clear_history_pressed)

                        if st.button("Clear History"):
                            write_text_data_file(username,"")
                            st.session_state.clear_history_pressed = True
                        
                        if previous_data:
                            if st.button("Process"):
                                with st.spinner("Processing"):
                                    previous_txt = read_previous_text(username)
                                    previous_txt = "\nThis is the previous text\n" + previous_txt
                                    write_text_data_file(username,previous_txt)
                                    text_chunks = get_text_chunks(previous_txt)
                                    vector_store = get_vector_store(text_chunks)
                                    st.session_state.conversation = get_conversational_chain(vector_store)
                                    st.success("Done")
                        else:
                            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
                            if st.button("Process"):
                                with st.spinner("Processing"):
                                    raw_text = get_text(pdf_docs)
                                    previous_txt = read_previous_text(username)
                                    previous_txt = raw_text + "\nThis is the previous text\n" + previous_txt
                                    write_text_data_file(username,previous_txt)
                                    st.session_state.clear_history_pressed = False
                                    text_chunks = get_text_chunks(previous_txt)
                                    vector_store = get_vector_store(text_chunks)
                                    st.session_state.conversation = get_conversational_chain(vector_store)
                                    st.success("Done")
                        Authenticator.logout('Log Out','sidebar')

                elif not authentication_status:
                    with info:
                        st.error('Invalid Username and Password')
                else:
                    with info:
                        st.warning('Please enter credentials')
            else:
                with info:
                    st.warning("Username does not exist, Please Sign up")

    
    except:
        st.success('Refresh Page')


if __name__ == "__main__":
    data_file = "data.obj"
    main()