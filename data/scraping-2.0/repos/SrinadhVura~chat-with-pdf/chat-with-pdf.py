import tempfile
from PIL import Image
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st 
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import *
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents.agent_toolkits import *
import base64
import requests
import json
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import base64
COGNITO_DOMAIN=st.secrets["COGNITO_DOMAIN"]
CLIENT_ID=st.secrets["CLIENT_ID"]
CLIENT_SECRET=st.secrets["CLIENT_SECRET"]
APP_URI=st.secrets["APP_URI"]

def initialise_st_state_vars():
    """
    Initialise Streamlit state variables.

    Returns:
        Nothing.
    """
    if "auth_code" not in st.session_state:
        st.session_state["auth_code"] = ""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user_cognito_groups" not in st.session_state:
        st.session_state["user_cognito_groups"] = []
def get_auth_code():
    """
    Gets auth_code state variable.

    Returns:
        Nothing.
    """
    auth_query_params = st.experimental_get_query_params()
    try:
        auth_code = dict(auth_query_params)["code"][0]
    except (KeyError, TypeError):
        auth_code = ""

    return auth_code


# ----------------------------------
# Set authorization code after login
# ----------------------------------
def set_auth_code():
    """
    Sets auth_code state variable.

    Returns:
        Nothing.
    """
    initialise_st_state_vars()
    auth_code = get_auth_code()
    st.session_state["auth_code"] = auth_code
def get_user_tokens(auth_code):
    token_url=f"{COGNITO_DOMAIN}/oauth2/token"
    client_secret_string=f"{CLIENT_ID}:{CLIENT_SECRET}"
    client_secret_encoded=str(base64.b64encode(client_secret_string.encode("utf-8")),"utf-8")
    headers={
        "Content-Type":"application/x-www-form-urlencoded",
        "Authorization":f"Basic {client_secret_encoded}",
    }
    body = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": auth_code,
        "redirect_uri": APP_URI,
    }

    token_response = requests.post(token_url, headers=headers, data=body)
    try:
        access_token = token_response.json()["access_token"]
        id_token = token_response.json()["id_token"]
    except (KeyError, TypeError):
        access_token = ""
        id_token = ""

    return access_token, id_token
def get_user_info(access_token):
    """
    Gets user info from aws cognito server.

    Args:
        access_token: string access token from the aws cognito user pool
        retrieved using the access code.

    Returns:
        userinfo_response: json object.
    """
    userinfo_url = f"{COGNITO_DOMAIN}/oauth2/userInfo"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {access_token}",
    }

    userinfo_response = requests.get(userinfo_url, headers=headers)

    return userinfo_response.json()


# -------------------------------------------------------
# Decode access token to JWT to get user's cognito groups
# -------------------------------------------------------
# Ref - https://gist.github.com/GuillaumeDerval/b300af6d4f906f38a051351afab3b95c
def pad_base64(data):
    """
    Makes sure base64 data is padded.

    Args:
        data: base64 token string.

    Returns:
        data: padded token string.
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += "=" * (4 - missing_padding)
    return data


def get_user_cognito_groups(id_token):
    """
    Decode id token to get user cognito groups.

    Args:
        id_token: id token of a successfully authenticated user.

    Returns:
        user_cognito_groups: a list of all the cognito groups the user belongs to.
    """
    user_cognito_groups = []
    if id_token != "":
        header, payload, signature = id_token.split(".")
        printable_payload = base64.urlsafe_b64decode(pad_base64(payload))
        payload_dict = json.loads(printable_payload)
        try:
            user_cognito_groups = list(dict(payload_dict)["cognito:groups"])
        except (KeyError, TypeError):
            pass
    return user_cognito_groups


# -----------------------------
# Set Streamlit state variables
# -----------------------------
def set_st_state_vars():
    """
    Sets the streamlit state variables after user authentication.
    Returns:
        Nothing.
    """
    initialise_st_state_vars()
    auth_code = get_auth_code()
    access_token, id_token = get_user_tokens(auth_code)
    user_cognito_groups = get_user_cognito_groups(id_token)

    if access_token != "":
        st.session_state["auth_code"] = auth_code
        st.session_state["authenticated"] = True
        st.session_state["user_cognito_groups"] = user_cognito_groups
login_link = f"{COGNITO_DOMAIN}/login?client_id={CLIENT_ID}&response_type=code&scope=email+openid&redirect_uri={APP_URI}"
logout_link = f"{COGNITO_DOMAIN}/logout?client_id={CLIENT_ID}&logout_uri={APP_URI}"
html_css_login = """
<style>
.button-login {
  background-color: skyblue;
  color: white !important;
  padding: 1em 1.5em;
  text-decoration: none;
  text-transform: uppercase;
}
.button-login:hover {
  background-color: #555;
  text-decoration: none;
}
.button-login:active {
  background-color: black;
}
</style>
"""
html_button_login = (
    html_css_login
    + f"<a href='{login_link}' class='button-login' >Log In</a>"
)
html_button_logout = (
    html_css_login
    + f"<a href='{logout_link}' class='button-login' >Log Out</a>"
)
def button_login():
    """
    Returns:
        Html of the login button.
    """
    return st.sidebar.markdown(f"{html_button_login}", unsafe_allow_html=True)
def button_logout():
    """
    Returns:
        Html of the logout button.
    """
    return st.sidebar.markdown(f"{html_button_logout}", unsafe_allow_html=True)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()) # Encodes the image to base 64
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()}); # Adds the background image to the app
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True # Allows the HTML code to be displayed
    )
#add_bg_from_local('page.png')    # Calls the function to add the background image
st.set_page_config(page_title="Chat and Listen PDF",page_icon="iconch.png",layout="wide",initial_sidebar_state="expanded")
img=Image.open("chat and listen pdf.png")
st.image(img)
st.title("📄 Chat and Listen PDF 🤖") # Title of the app
st.write("Will build a conversational AI bot right on the fly from the PDF provided by you. Just upload the PDF and I will learn it. I can answer all your questions in a chiffy. I can speak out your answers too")
# st.set_page_config(page_icon="🤖")
set_st_state_vars()
#Adding buttons
if st.session_state["authenticated"]:
    button_logout()
else:
    button_login()

if  (st.session_state["authenticated"]):
    st.sidebar.title("Give me the PDF to learn") # Title of the sidebar
    upload=st.sidebar.file_uploader("Upload your PDF",type=['pdf']) # File uploader
    if upload is not None:
        tdir=tempfile.TemporaryDirectory()  # Temporary directory to store the uploaded file
        tpath=os.path.join(tdir.name,'file.pdf') 
        with open(tpath,'wb') as f:
            f.write(upload.getbuffer()) # Writes the uploaded file to the temporary directory
        st.sidebar.write("Uploaded successfully")
        # Message to be displayed after the file is uploaded
    
    
    # os.environ['OPENAI_API_KEY']
    
        model=OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],model="gpt-3.5-turbo") # Model to be used for the bot. You can change it to any other model from the list of models supported by OpenAI to get different results.
        emb=OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])  # Embeddings to be used for the bot.
        load=PyPDFLoader(tpath) # Loader to load the PDF file
        pages=load.load()      # Loads the PDF file
        splitter=TokenTextSplitter(chunk_size=1000,chunk_overlap=0)
        split_data=splitter.split_documents(pages)    # Splits the PDF file into chunks of 1000 tokens each
        vectDB = FAISS.from_documents(split_data,
                              emb,
                              )              # Creates the vector database from the PDF file and stores it in the local directory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Creates a memory for the bot to remember the conversation
        chatQA = ConversationalRetrievalChain.from_llm(
                    OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],
                       temperature=0, model_name="gpt-3.5-turbo"), 
                    vectDB.as_retriever(), 
                    memory=memory)               # Creates the bot
        
        chat_history=[]
        prompt=st.text_input("Enter your prompt")      # Takes the prompt from the user
        if prompt:
                with st.spinner('Generating response...'):                   
                    response = chatQA({"question": prompt+"limit your answer to less than 50 words" ,"chat_history":chat_history}, return_only_outputs=True) # Generates the response from the bot
                    answer = response['answer']
                    st.write(answer)        # Displays the response
                    myobj = gTTS(text=answer,lang='en', slow=False)     # Converts the response to speech
                    mp3_play=BytesIO()         # Creates a BytesIO object
                    myobj.write_to_fp(mp3_play)
                    st.audio(mp3_play,format="audio/mp3")      # Plays the audio
        else:
                    st.warning('Please enter your prompt')
else:
    if st.session_state["authenticated"]:
        st.write("No access, please check your credentials.")
    else:
        st.write("Please Login!")
