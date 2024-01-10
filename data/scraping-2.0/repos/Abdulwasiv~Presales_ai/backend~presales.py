from tempfile import NamedTemporaryFile
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


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

def handle_csv_agent(file_content):
    agent = create_csv_agent(
        OpenAI(temperature=0), file_content, verbose=True)
    user_question = st.text_input("Ask a question about your CSV: ")
    if user_question:
        with st.spinner(text="In progress..."):
            st.write(agent.run(user_question))

def handle_pdf_agent(file_content):
    # get pdf text
    raw_text = get_pdf_text(file_content)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question,"conversation_history.pdf")

def handle_userinput(user_question,pdf_output_path):
    response = st.session_state.conversation({'question': user_question})
    # st.session_state.chat_history = response['chat_history']

     # Append the conversation history to the PDF
    with open(pdf_output_path, 'a') as pdf_file:
        for i, message in enumerate(response['chat_history']):
            if i % 2 == 0:
                pdf_file.write(f"User: {message.content}\n")
            else:
                pdf_file.write(f"Bot: {message.content}\n")



    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main(pdf_output_path):
    load_dotenv()
    st.set_page_config(page_title="Presales AI", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("PRESALES AI 📈")

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        return

    files = st.file_uploader("Upload CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True)
    if files:
        csv_files = []
        pdf_files = []

        for file in files:
            file_extension = os.path.splitext(file.name)[1]
            if file_extension == '.csv':
                csv_files.append(file)
            elif file_extension == '.pdf':
                pdf_files.append(file)
            else:
                st.warning(f"Unsupported file type for {file.name}. Please upload a CSV or PDF file.")

        if csv_files:
            for csv_file in csv_files:
                with NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(csv_file.read())
                    temp_file.seek(0)  # Move the cursor to the beginning of the file
                    handle_csv_agent(temp_file.name)

        if pdf_files:  # Move the cursor to the beginning of the file
                    handle_pdf_agent(pdf_files)

if __name__ == "__main__":
    main("conversation_history.pdf")
