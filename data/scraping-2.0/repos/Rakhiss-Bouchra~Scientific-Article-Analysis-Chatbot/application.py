import streamlit as st
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
from transformers import pipeline
import PyPDF2  # For extracting text from PDF files
from PIL import Image


# Inclure les styles CSS personnalisés pour un thème sombre
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
            
def handle_userinput1(user_question, user_answer):
    # Assuming you have a response dictionary with 'chat_history' like OpenAI's model
    # Replace this with your own logic to generate a response
    response = {'chat_history': []}
    
    # Append the user's question and answer to the chat history
    response['chat_history'].append({'role': 'system', 'content': 'You:'})
    response['chat_history'].append({'role': 'user', 'content': user_question})
    response['chat_history'].append({'role': 'assistant', 'content': user_answer})

    # Update the session's chat history
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)


# Define the Hugging Face model checkpoint
model_checkpoint = "Zouhair1/bert-finetuned-squad-accelerate"
question_answerer = pipeline("question-answering", model=model_checkpoint)

def extract_text_fromPDF(pdf_docs):
    # Iterate through the list of uploaded PDF files
    extracted_text = ""
    for pdf_file in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()
    return extracted_text
def main():
    load_dotenv()
    st.set_page_config(page_title="Your Document Companion",
                       page_icon="📚🤖")

    st.write(css, unsafe_allow_html=True)
    
    # Opening the image
    image = Image.open(r"docs\logoof3d.png")

    # Displaying the image in the side bar of the Streamlit app
    st.sidebar.image(image)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # En-tête avec le nom du chatbot
    st.title("Welcome to DocuBot 📄🤖")

    # Add a radio button to choose between models
    selected_model = st.radio("Choose a Model:", ["OpenAI Model", "Hugging Face Model"])
    with st.sidebar:
        st.subheader("Analyzing Documents")
        pdf_docs = st.file_uploader(
            "Discover more from your PDFs. Upload your files, analyze with a click, and get ready for interactive discussions.", accept_multiple_files=True)
        if st.button("Analyze"):
            if pdf_docs is None or len(pdf_docs) == 0:
                st.write("Please upload a PDF file.")
            else:
                with st.spinner("Analyzing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)

    if selected_model == "OpenAI Model":
        # Use OpenAI model
        user_question = st.text_input("Hello, I'm DocuBot 🤖! How can I assist you with your documents today?")
    
        if st.session_state.conversation is not None:
            if user_question:
                handle_userinput(user_question)
        else:
            if user_question:
                st.write("Please upload a file and analyze it before asking questions.")
    elif selected_model == "Hugging Face Model":
        # Use Hugging Face model
        user_question = st.text_input("Hello, I'm DocuBot 🤖! How can I assist you with your documents today?")
        if st.session_state.conversation is not None:
            if user_question:    
                answer = question_answerer(question=user_question, context=extract_text_fromPDF(pdf_docs))
                handle_userinput1(user_question, answer["answer"])   
        else:
            if user_question:
                st.write("Please upload a file and analyze it before asking questions.")
if __name__ == '__main__':
    main()