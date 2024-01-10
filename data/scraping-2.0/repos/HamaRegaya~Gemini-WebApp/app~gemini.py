import os
import streamlit as st
import requests
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import google.auth
from google.cloud import storage, aiplatform
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import google.generativeai as genai 
import google.ai.generativelanguage as glm 
from PIL import Image
import io 
from gtts import gTTS  # new import

import base64
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    audio_bytes = tts.save("output.wav")
    with open("output.wav", "rb") as audio_file:
        base64_encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    return base64_encoded_audio



load_dotenv()
key_path = "GoogleCloudSecret.json"


# Set the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
openai = OpenAI()


tru = Tru()
credentials = Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

if credentials.expired:
    credentials.refresh(Request())

PROJECT_ID = "primeval-gizmo-406722"
REGION = "us-central1"
aiplatform.init(project=PROJECT_ID, location=REGION,credentials=credentials)




template = """
You are a professional customer support specialist chatbot, dedicated to providing helpful, accurate, and polite responses. 
Your goal is to assist users with their queries to the best of your ability. 
If a user asks something outside of your knowledge, politely inform them that you 
don't have the information they need and, if possible, suggest where they might find it. 
Remember to always maintain a courteous and supportive tone.

{chat_history}
Human: {human_input}
Chatbot:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_chain= LLMChain(
            llm=ChatGoogleGenerativeAI(model="gemini-pro",max_output_tokens=2048,temperature=0.9,top_p=1),
            memory=memory,
            prompt=prompt,
            verbose=True,
        )
conversation_contexts = {}



# Question/answer relevance between overall question and answer.
f_relevance = Feedback(openai.relevance).on_input_output()

# Moderation metrics on output
f_hate = Feedback(openai.moderation_hate).on_output()
f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

# TruLens Eval chain recorder
chain_recorder = TruChain(
    conversation_chain, app_id="contextual-chatbot", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
)



# Define your page functions
def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page!")

def chat_page():
    # Streamlit frontend
    st.title("Contextual Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Record with TruLens
            with chain_recorder as recording:
                full_response = conversation_chain.run(prompt)
            message_placeholder = st.empty()
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
        base64_encoded_audio = text_to_speech(full_response)
        with open("speaker.html", "r") as html_file:
                html_content = html_file.read().replace("{{base64_encoded_audio}}", base64_encoded_audio)
        st.markdown(html_content, unsafe_allow_html=True)
        
# ########################################

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

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            


    



def document_page():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
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

# #######################################################################
def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr
                
def chat_image():
        st.header("Interact with Gemini Pro Vision")
        st.write("")

        image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
        uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), use_column_width=True)

            st.markdown("""
                <style>
                        img {
                            border-radius: 10px;
                        }
                </style>
                """, unsafe_allow_html=True)
            
        if st.button("GET RESPONSE", use_container_width=True):
            model = genai.GenerativeModel("gemini-pro-vision")

            if uploaded_file is not None:
                if image_prompt != "":
                    image = Image.open(uploaded_file)

                    response = model.generate_content(
                        glm.Content(
                            parts = [
                                glm.Part(text=image_prompt),
                                glm.Part(
                                    inline_data=glm.Blob(
                                        mime_type="image/jpeg",
                                        data=image_to_byte_array(image)
                                    )
                                )
                            ]
                        )
                    )

                    response.resolve()

                    st.write("")
                    st.write(":blue[Response]")
                    st.write("")

                    st.markdown(response.text)
                    base64_encoded_audio = text_to_speech(response.text)
                    with open("speaker.html", "r") as html_file:
                            html_content = html_file.read().replace("{{base64_encoded_audio}}", base64_encoded_audio)
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.write("")
                    st.header(":red[Please Provide a prompt]")

            else:
                st.write("")
                st.header(":red[Please Provide an image]")
# ########################################################################
def contact_page():
    st.title("Contact Page")
    st.write("Feel free to contact us.")
    

# Create a dictionary to map page names to page functions
pages = {
    "Home": home_page,
    "Chat": chat_page,
    "Chat with multiple Document":document_page,
    "Chat with image":chat_image,
    "Contact": contact_page,
}

# Create a sidebar with menu items
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# Display the selected page
pages[selected_page]()


tru.run_dashboard()