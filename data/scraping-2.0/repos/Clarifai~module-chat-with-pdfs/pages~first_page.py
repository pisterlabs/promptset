import streamlit as st
import tempfile
import time
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Clarifai
from langchain.chains import RetrievalQA
from clarifai.modules.css import ClarifaiStreamlitCSS

st.title("🦜 Chat with your PDF Files")

ClarifaiStreamlitCSS.insert_default_css(st)

@st.cache_resource(ttl="1h")
def load_chunk_pdf(uploaded_files):
    # Read documents
    documents = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def vectorstore(USER_ID, APP_ID, docs, CLARIFAI_PAT):
    clarifai_vector_db = Clarifai.from_documents(
        user_id=USER_ID,
        app_id=APP_ID,
        documents=docs,
        pat=CLARIFAI_PAT,
        number_of_docs=3,
    )
    return clarifai_vector_db


def QandA(CLARIFAI_PAT, clarifai_vector_db):
    from langchain.llms import Clarifai
    USER_ID = "openai"
    APP_ID = "chat-completion"
    MODEL_ID = "GPT-4"

    # completion llm
    clarifai_llm = Clarifai(
        pat=CLARIFAI_PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)

    qa = RetrievalQA.from_chain_type(
        llm=clarifai_llm,
        chain_type="stuff",
        retriever=clarifai_vector_db.as_retriever()
    )
    return qa


def main():
    with st.sidebar:
        st.subheader("Add your Clarifai PAT, USER ID, APP ID along with the documents")

        # Get the USER_ID, APP_ID, Clarifai API Key
        CLARIFAI_PAT = st.text_input("Clarifai PAT", type="password")
        USER_ID = st.text_input("Clarifai user id")
        APP_ID = st.text_input("Clarifai app id")

        uploaded_files = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)

    if not (CLARIFAI_PAT and USER_ID and APP_ID and uploaded_files):
        st.info("Please add your Clarifai PAT, USER_ID, APP_ID and upload files to continue.")
    else:
        # process pdfs
        docs = load_chunk_pdf(uploaded_files)

        # create a vector store
        clarifai_vector_db = vectorstore(USER_ID, APP_ID, docs, CLARIFAI_PAT)

        # create Q&A chain
        conversation = QandA(CLARIFAI_PAT, clarifai_vector_db)

        # Ask the question to the GPT 3.5 Turbo model based on the documents

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Chat with your PDF Files?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner(text="Getting the Response"):
                    response = conversation.run(prompt)
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + " ")
                message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()