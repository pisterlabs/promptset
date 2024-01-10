import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
from langchain.llms import OpenAIChat
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from UI.htmlTemplates import css, bot_template, user_template
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms import HuggingFaceHub
import textwrap
import os
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document



openai_api_key=''
os.environ["OPENAI_API_KEY"] = openai_api_key
load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        #separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


persist_directory = '/Users/Nian/Desktop/School/BusinessSchool_RA/LLMs/vdb'

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    vectorstore = Chroma.from_documents(documents=documents, 
                                        embedding=embeddings, 
                                        persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

def get_conversation_chain(vectorstore):
    vectorstore = Chroma(persist_directory=persist_directory, 
                         embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key))

    llm = OpenAIChat(temperature=0.5, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                               retriever=vectorstore.as_retriever(), 
                                                               memory=memory)
    return conversation_chain



def Summarize_Document(text):
    if(len(text)):
        print("***Text Contents***")
        print(text)
        #llm = OpenAIChat(temperature=0.5, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key) 
        llm = OpenAI(model_name="text-davinci-003")
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
        #separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
        texts = text_splitter.split_text(text)
        # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
        # Text summarization
        SummarizationChain = load_summarize_chain(llm, chain_type="map_reduce", verbose = True)
        output_summary = SummarizationChain.run(docs)
        wrapped_text = textwrap.fill(output_summary, width=120)
        print(wrapped_text)
    return wrapped_text


def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history.extend(response['chat_history'])

        for message in response['chat_history']:
            if message.type == 'user':
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Conversation chain not initialized. Please upload and process your documents first.")

            


def main():
    st.set_page_config(page_title="KFBS Chatbot", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Display previous conversation
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message.type == 'user':
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    st.header("KFBS Chatbot:books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
        if st.session_state.vectorstore is not None:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_question)

            # You can then display the retrieved documents, for example:
            for doc in docs:
                st.write(doc.page_content)
        else:
            st.write("No documents have been processed yet.")

    with st.sidebar:
        image = Image.open('UI/Logo_of_UNC_Kenan–Flagler_Business_School (1).png')
        st.image(image)

        st.subheader("Your documents")

        # Checking if vectorstore is loaded
        if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    st.session_state.vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

if __name__ == '__main__':
    main()
