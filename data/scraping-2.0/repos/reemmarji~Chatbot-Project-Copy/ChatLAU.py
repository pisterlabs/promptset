import streamlit as st
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from cryptography.fernet import Fernet


with open("key.key", "rb") as key_file:
    key =  key_file.read()
with open("encrypted.key", "rb") as encrypted_message:
    encrypted_message =  encrypted_message.read()

fernet = Fernet(key)
decrypted_message = fernet.decrypt(encrypted_message)
OPENAI_API_KEY = decrypted_message.decode()


st.markdown(
    """
    <style>
    .title {
        color: white; /* Set text color to white */
        font-size: 75px; /* Set font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="title">ChatLAU</h1>', unsafe_allow_html=True)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://reemsbucket.s3.us-west-2.amazonaws.com/background.jpeg");
             background-attachment: fixed;
             background-size: cover;
             color: white;  /* Set text color to white */
         }}
         </style>  
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.markdown(
    """
    <style>
    .text-input-label {
        color: white; /* Set text color to white */
        margin-bottom: 0; /* Remove bottom margin */
        font-size: 18px; /* Set font size to 18 pixels */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st_version = str(st.__version__)
st.write('<p class="text-input-label">Tell me something...</p>', unsafe_allow_html=True)

css = '''
    <style>
        .text-container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            margin-bottom: 10px;
            color: black; /* Set the text color to black */
        }

        .text-container span {
            color: black;
        }
    </style>
'''

st.markdown(css, unsafe_allow_html=True)

try:
    index_name = 'v1-index-pinecone'
    text_field = "text"
    question = st.text_input('', value='', key=None, type='default', help=None)

    index = pinecone.Index(index_name)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pinecone.init(api_key="681591b9-096b-4da3-8df2-9de5f89fba34", environment="northamerica-northeast1-gcp")

    vectorstore = Pinecone(
        index, embeddings.embed_query, text_field
    )

    query = question

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    response = qa.run(query)
    if question:
        st.markdown(f'<div class="text-container"><span>You:</span> {question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="text-container"><span>ChatLAU:</span> {response}</div>', unsafe_allow_html=True)

except Exception as e:
    response = None