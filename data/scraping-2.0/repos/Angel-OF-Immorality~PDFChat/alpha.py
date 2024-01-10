import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from templates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def main():
    load_dotenv()
    st.set_page_config(page_title="Document Summarizer", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Doc")
    user_question = st.text_input("Ask me Questions about your Docs")
    if user_question:
        handle_user_input(user_question)


    with st.sidebar:
        st.subheader("Your Docs")
        docs = st.file_uploader("Upload PDFs Here & Press Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #Get PDFs and Convert it to Text Chunks
                raw_text = get_pdf_text((docs))
                #get the text Chunks
                text_chunks = get_text_chunks(raw_text)
                #Creating Embeddings 
                #By using OpenAI Model (Paid)
                #Creating Vector Stores
                vector_store = get_vector_store(text_chunks)
                #Creating instance of Conversation Chain
                st.session_state.conversation = get_conv_chain(vector_store)  


def get_pdf_text(pdfs):
    text = "" 
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore 

def get_conv_chain(vectorstore):
    # chat_model = ChatOpenAI()
    chat_model = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
    converation_chain = ConversationalRetrievalChain.from_llm(
        llm = chat_model,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return converation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question' : user_question})
    st.session_state.chat_history = response('chat_history')
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()