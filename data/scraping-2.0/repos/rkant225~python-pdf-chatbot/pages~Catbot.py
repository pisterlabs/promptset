import streamlit as st
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext, load_index_from_storage
import os
from pathlib import Path
import openai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Text to speech imports
# from gtts import gTTS
# from pygame import mixer
# from googletrans import Translator

st.set_page_config(layout="centered")
ss = st.session_state

openai.api_key = os.getenv("OPENAI_API_KEY")

# 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 - Rahul K. Key
# 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 - Rahul S. Key

# Translator INIT
# translator = Translator()
# mixer.init()



# ============================ Authenticator - START ==============================
with open('./auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
# ============================ Authenticator - END ==============================





# ======================= Admin helper methods - START ===================================

def loadPdfsAndGenerateIndex():
    with st.spinner("Creating Index from Train documents...."):
        documents = SimpleDirectoryReader('./train_docs').load_data()
        index = GPTVectorStoreIndex(documents)
        index.storage_context.persist(persist_dir="./index")

def upload(uploaded_file):
    cwd = os.getcwd()
    completePath = Path(cwd, "train_docs", uploaded_file.name)
    with open(completePath, mode='wb') as w:
        w.write(uploaded_file.getvalue())


def updateFileList():
    cwd = os.getcwd()
    fileNames = os.listdir(Path(cwd, "train_docs"))
    st.session_state.file_List = fileNames

def renderFileList():
    # listHTMLString = ""
    cwd = os.getcwd()
    fileNames = os.listdir(Path(cwd, "train_docs"))
    # for file in fileNames:
    #     listHTMLString = listHTMLString + "<li>" + file + "<button> x </button>" + "</li>"
    # st.markdown(listHTMLString, unsafe_allow_html=True)
   
    for index,file in enumerate(fileNames):
        cwd = os.getcwd()
        emp = st.empty()

        col1, col2, col3 = st.columns([7, 2, 3])
        col1.markdown('<b>'+ file + '</b>', unsafe_allow_html=True)

        currentFilePath = Path(cwd, "train_docs", file)
        if col2.button("Del", key=index):
            if os.path.isfile(currentFilePath):
                os.remove(currentFilePath)
                st.experimental_rerun()

        if os.path.isfile(currentFilePath):
            with open(currentFilePath, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
                col3.download_button(label="Download", data=PDFbyte, file_name=file, mime='application/octet-stream')
            
                

        # else:
        #     emp.empty()
# ======================= Admin helper methods - END ===================================





# ======================= Normal User helper methods - START ===================================

# languageDictonary = {
#     'English': 'en',
#     'Hindi': 'hi',
#     'Arabic': 'ar',
#     'Assamese': 'as',
#     'Bengali': 'bn',
#     'Chinese': 'zh',
#     'French': 'fr',
#     'German': 'de',
#     'Gujarati': 'gu',
#     'Kannada': 'kn',
#     'Tamil': 'ta',
#     'Telugu': 'te',
# }

# Set initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]


# Helper method to generate audio and play it
# def createAudioFromTextAndPlayIt(text):

#     # Remove the old audio file if it exists
#     if os.path.isfile('assistant.mp3'):
#         mixer.music.unload()
#         os.remove("assistant.mp3")

#     audio = gTTS(text=text, lang=languageDictonary[st.session_state.assistant_language])
#     audio.save("assistant.mp3")
#     mixer.music.load("assistant.mp3")
#     mixer.music.play()

#     # Uncomment below "while" loop if you want to wait for audio to be finished.
#     # while mixer.music.get_busy():
#     #     continue


# def stopAudioPlayback():
#     mixer.music.stop()

name, authentication_status, username = authenticator.login('RockWell Automation ChatBot Login', 'main')

if st.session_state.authentication_status == False:
    st.error('Username/password is incorrect')
    
# If user is not logged in then ask them to login first
if not st.session_state.authentication_status:
    st.error("You must log-in to chat with our chatbot!")
    st.stop()  # App won't run anything after this line

# ======================= Normal User helper methods - START ===================================







# ======================================== UI - START =========================================

# Sidebar
with st.sidebar:
    # App title
    st.title('Rockwell Automation')

    # User detail and logout button
    st.header('Welcome "' + st.session_state.name + '"!')
    authenticator.logout('Logout', 'sidebar')

    # UI selection
    selectedUi_options = ["Chatbot", "Chat History", "User Details"]
    if st.session_state.username == "admin":
        selectedUi_options.append("Admin UI")
    st.radio("Selected UI", selectedUi_options ,key="selectedUi",horizontal=False)

    # st.selectbox(label="Text output and playback language", options=['English', 'Hindi', 'Arabic', 'Assamese', 'Bengali', 'Chinese', 'French', 'German', 'Gujarati', 'Kannada', 'Tamil', 'Telugu'], key="assistant_language")

    # st.checkbox('Do you want your assistant to read out the answers?', key="with_text_to_speach")
    # st.button('Stop assistant playback', on_click=stopAudioPlayback, disabled=not st.session_state.with_text_to_speach)

# Chatbot UI
if st.session_state.selectedUi == "Chatbot":
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    help_text = ""
    if os.path.isdir('./index') and os.path.isfile("./index/docstore.json") and os.path.isfile("./index/graph_store.json") and os.path.isfile("./index/index_store.json") and os.path.isfile("./index/vector_store.json"):
        help_text = "Enter your query..."
    else:
        help_text = "Chatbot is not yet ready, please connect with Admin."

    # User Input
    if prompt := st.chat_input(disabled=not (os.path.isdir('./index') and os.path.isfile("./index/docstore.json") and os.path.isfile("./index/graph_store.json") and os.path.isfile("./index/index_store.json") and os.path.isfile("./index/vector_store.json")), placeholder=help_text):
        st.session_state.messages.append({"role": "user", "content": prompt})
        ss['query'] = prompt
        with st.chat_message("user"):
            st.write(prompt)


    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                loaded_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./index"))
                query_engine = loaded_index.as_query_engine()
                eng_response = query_engine.query(ss['query']).response # English
                
                # translated_text = translator.translate(eng_response, dest=languageDictonary[st.session_state.assistant_language]) # Translated
                # response = translated_text.text
                # st.write(response)
                st.write(eng_response)

        message = {"role": "assistant", "content": eng_response}
        st.session_state.messages.append(message)

        # Play the sound
        # if st.session_state.with_text_to_speach:
        #     createAudioFromTextAndPlayIt(response)

    # Delete the mp3 file
    # if not mixer.music.get_busy() and os.path.isfile('assistant.mp3'):
        # mixer.music.unload()
        # os.remove("assistant.mp3")



# Chat History UI
if st.session_state.selectedUi == "Chat History":
    st.header('Chat history')
    allChats = st.session_state.messages

    questions = ["Hi"]
    answers = []

    for chat in allChats:
        if chat["role"] == "assistant":
            answers.append(chat["content"])
        if chat["role"] == "user":
            questions.append(chat["content"])

    tableRows = ""
    for index in range(0, len(questions)):
        tableRows = f"""
                        <tr>
                            <td>{questions[index]}</td>
                            <td>{answers[index]}</td>
                        </tr>
                    """
        
        table = f"""
                    <table>
                        <tr>
                            <th>Question</th>
                            <th>Answer</th>
                        </tr>{tableRows}
                    </table>
                """
        st.markdown(table, unsafe_allow_html=True)


# User Details UI
if st.session_state.selectedUi == "User Details":
    st.header('User Details')
    userDetailsMarkdown = f""" 
        ### Name
            {st.session_state.name}
        ### Username
            {st.session_state.username}
        ### IsAuthenticated
            {st.session_state.authentication_status}
    """
    st.markdown(userDetailsMarkdown)


# Admin UI
if st.session_state.selectedUi == "Admin UI":
    st.title('Rockwell Automation (Admin)')

    st.header('Welcome "' + st.session_state.name + '"!')
    # authenticator.logout('Logout', 'main')

    st.header("1. Select and save your new User manual to training folder.")

    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        submitted = st.form_submit_button("Store this file to '/train_docs' for training")
        if submitted and uploaded_file:
            upload(uploaded_file)

    # uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    # if uploaded_file:
    #     upload(uploaded_file)

    # st.button("Store this file to '/train_docs'", on_click=upload, disabled=not uploaded_file)

    st.header("2. List of current training documents/manual.")
    updateFileList()
    renderFileList()

    st.header("3. Finaly generate index out of training documents.")
    st.markdown("<i>The button located below serves a crucial function in our system - it's the key to generating an index from your PDF documents. By clicking on this button, you initiate a process that will extract essential information and organize it into a comprehensive index. This feature can save you valuable time and effort when you need to quickly access specific content within your PDFs. So, whenever you're ready to create an index from your documents, simply click on the button below to get started!</i>", unsafe_allow_html=True)
    st.button('Generete index out of latest documents', on_click=loadPdfsAndGenerateIndex)


# ======================================== UI - START =========================================