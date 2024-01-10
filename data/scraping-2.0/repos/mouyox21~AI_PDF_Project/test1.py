import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import sqlite3
import hashlib

st.set_page_config(page_title="Application de Chatbot", layout="wide", page_icon=":books:")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background: white;
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background: rgb(2,0,36);
background: linear-gradient(245deg, rgba(2,0,36,1) 0%, rgba(9,107,121,1) 35%, rgba(0,212,255,1) 100%);
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Connect to the database
conn = sqlite3.connect('user_credentials.db')
c = conn.cursor()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Return the content of the response message as a string
    return response['chat_history'][-1].content 



# Fonction pour la page de connexion
def login_page():
    with st.sidebar:
        st.title('🤗💬 Pdf Profiler')
        st.markdown('''
        ## About
        Une application développée dans le cadre de **mon projet de fin d'études en troisième année**. 
        Cette application est un chatbot alimenté par l'**IA** (Intelligence Artificielle) LLM et offre des fonctionnalités puissantes pour l'analyse et le profilage de fichiers **PDF**.
        
        ## Fonctionnalités principales :

        - Analyse de PDF en profondeur
        - Extraction de métadonnées
        - Recherche intelligente
        - Visualisation des données
        - Analyse comparative
    
        ''')
        st.write('Made with ❤️ by [Mouad ANIBA](#)')
    st.title("Page de Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se Connecter"):
        # Hash the entered password for comparison with the stored hash
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check if the username and hashed password match a record in the database
        c.execute("SELECT id, username FROM users WHERE username=? AND password=?", (username, hashed_password))
        user = c.fetchone()

        if user:
            st.success("Connecté en tant que {}".format(username))
            # Crée une session pour l'utilisateur connecté
            st.session_state.logged_in = True
            st.session_state.username = username
            # Redirige vers la page suivante
            st.experimental_rerun()
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")

    if st.button("S'inscrire"):
        st.session_state.signup = True
        st.experimental_rerun()
# Fonction pour la page principale après la connexion

def signup_page():
    with st.sidebar:
        st.title('🤗💬 Pdf Profiler')
        st.markdown('''
        ## About
        Une application développée dans le cadre de **mon projet de fin d'études en troisième année**. 
        Cette application est un chatbot alimenté par l'**IA** (Intelligence Artificielle) LLM et offre des fonctionnalités puissantes pour l'analyse et le profilage de fichiers **PDF**.
        
        ## Fonctionnalités principales :

        - Analyse de PDF en profondeur
        - Extraction de métadonnées
        - Recherche intelligente
        - Visualisation des données
        - Analyse comparative
    
        ''')
        st.write('Made with ❤️ by [Mouad ANIBA](#)')

    st.title("Page d'Inscription")
    new_username = st.text_input("Nouveau nom d'utilisateur")
    new_password = st.text_input("Nouveau mot de passe", type="password")
    new_avatar = st.file_uploader("Choisir un avatar", type=["jpg", "png"])

    if st.button("S'inscrire"):
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        
        # Save the avatar file
        if new_avatar:
            avatar_path = f"avatars/{new_username}_{new_avatar.name}"
            with open(avatar_path, "wb") as f:
                f.write(new_avatar.read())
        else:
            avatar_path = None
        
        # Insert the new user's information into the database
        try:
            c.execute("INSERT INTO users (username, password, avatar) VALUES (?, ?, ?)",
                      (new_username, hashed_password, avatar_path))
            conn.commit()
            st.success("Inscription réussie ! Connectez-vous maintenant.")
        except sqlite3.IntegrityError:
            st.error("Ce nom d'utilisateur existe déjà.")

    if st.button("Retour"):
        st.session_state.signup = False  # Réinitialise la session d'inscription
        st.experimental_rerun()

def intialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about"]
    if "past" not in st.session_state:
        st.session_state['past'] = ['Hey!']

def display_chat_histroy():
    reply_container = st.container()
    container =st.container()
    
    with container:
        with st.form(key='my_form',clear_on_submit=True):
            user_question = st.text_input("Question:",placeholder="Ask me about resources",key='input')
            submit_button = st.form_submit_button(label='send')
        
        if submit_button and user_question:
            output = handle_userinput(user_question)
            st.session_state['past'].append(user_question)
            st.session_state['generated'].append(output)
    
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state['past'][i],is_user=True,key=str(i)+"_user",avatar_style="thumbs")
                message(st.session_state['generated'][i],key=str(i),avatar_style="pixel-art")
def main_page():
    
    # Contenu de la page principale (chatbot, paramètres, etc.)
    # Ajoutez ici les éléments souhaités
    
    
    load_dotenv()
    
    st.write(css, unsafe_allow_html=True)

    intialize_session_state()

    st.title("Chat with multiple PDFs :books:")
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
                # get pdf text
            raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
            text_chunks = get_text_chunks(raw_text)

                # create vector store
            vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    
    

    

    display_chat_histroy()
    with st.sidebar:
        st.title('🤗💬 Pdf Profiler')

        
        st.markdown('''
        ## About
        Une application développée dans le cadre de **mon projet de fin d'études en troisième année**. 
        Cette application est un chatbot alimenté par l'**IA** (Intelligence Artificielle) LLM et offre des fonctionnalités puissantes pour l'analyse et le profilage de fichiers **PDF**.
        
        ## Fonctionnalités principales :

        - Analyse de PDF en profondeur
        - Extraction de métadonnées
        - Recherche intelligente
        - Visualisation des données
        - Analyse comparative
    
        ''')

        st.write('Made with ❤️ by [Mouad ANIBA](#)')
                
        if st.button("Déconnexion"):
        # Supprime la session de l'utilisateur et retourne à la page de connexion
            st.session_state.logged_in = False
            st.experimental_rerun()
 
    
# Configuration de l'application Streamlit
def main():
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # Vérifie si l'utilisateur est connecté
    if hasattr(st.session_state, 'logged_in') and st.session_state.logged_in:
        main_page()
    elif hasattr(st.session_state, 'signup') and st.session_state.signup:
        signup_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
