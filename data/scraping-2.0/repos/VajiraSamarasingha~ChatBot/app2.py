import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

import os

load_dotenv()

chat_style = """
<style>
.chat-container {
    background-color: #f0f0f0; /* Set your desired background color */
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
}

.message-user {
    background-color: #0074cc; /* User message background color */
    color: white;
    border-radius: 5px;
    padding: 5px 10px;
    margin: 5px;
}

.message-panda {
    background-color: #ff9900; /* Chatbot message background color */
    color: white;
    border-radius: 5px;
    padding: 5px 10px;
    margin: 5px;
}

.message-input {
    background-color: white; /* Message input background color */
    border: none;
    border-radius: 20px;
    padding: 10px 20px;
    margin-top: 10px;
    width: 80%;
}

.send-button {
    background-color: #0074cc; /* Send button background color */
    color: white;
    border: none;
    border-radius: 50%;
    padding: 10px;
    margin-top: 10px;
    margin-left: 5px;
    cursor: pointer;
}
</style>
"""

st.markdown(chat_style, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello, Ask me anything"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]

def conversational_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Document", key="input")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner("Generate response ......."):
                output = conversational_chat(user_input, chain, st.session_state["history"])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + "_user", avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="buddhist")

def create_conversational_chain(vector_store):
    load_dotenv()
    # create llm
    llm = Replicate(
        streaming=True,
        model="meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
        input={"temperature": 0.1, "max_length": 500, "top_p": 1}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)

    return chain

def main():
    initialize_session_state()
    st.title("Buddhist AI Chatbot")

    # Specify the PDF file path
    pdf_file_path = "tipitaka.pdf"  # Replace with the actual PDF file path

    if os.path.exists(pdf_file_path):
        # Load the PDF file directly using PyPDFLoader
        loader = PyPDFLoader(pdf_file_path)
        text = loader.load()
    else:
        st.error("PDF file not found. Make sure to provide the correct PDF file path.")
        return

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # create embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                          model_kwargs={"device": "cpu"})

    # create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()
