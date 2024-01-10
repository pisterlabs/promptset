import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import glob


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)

    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):

    llm = ChatOpenAI(temperature=0.75)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
    #                     model_kwargs={"temperature": 0.25, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.set_page_config(
        page_title="Chatea con nuestro experto f", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatea con nuestro experto :books:")

    user_question = st.text_input(
        "Haz una pregunta acerca de nuestra base de conocimiento en los documentos")

    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "Hola Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "Hola Humano"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Los documentos")

        pdf_docs = glob.glob('./reglamentos/*.pdf')

        if st.button("Proceso"):
            with st.spinner("Procesando, paciencia por favor"):

                # get pdf text:
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector database / vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chain

                # conversation = get_conversation_chain(vector_store)
                st.session_state.conversation = get_conversation_chain(
                    vector_store)


if __name__ == '__main__':
    main()
