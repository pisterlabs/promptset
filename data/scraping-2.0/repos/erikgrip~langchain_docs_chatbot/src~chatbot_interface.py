import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.openai import OpenAI

from src.doc_store import DocStore
from src.utils.data import download_and_unzip

REPO_ZIP_URL = "https://github.com/langchain-ai/langchain/archive/refs/heads/master.zip"
TARGET_EXTENSIONS = [".md", ".mdx"]


def streamlit_chatbot_app(
    force_data_download, delete_persisted_db, num_retrieved_docs, temperature
):
    """
    Run a Streamlit chatbot interface using a langchain retrieval chain.

    Parameters:
    -----------
    qa_chain : Chain
        A conversational retrieval model that takes a question and a chat history
        as input, and returns an answer as output.

    Returns:
    --------
    None
    """
    st.title("🦜️🔗 Langchain Docs Chatbot 🤖")

    # Initialize chat history
    if not st.session_state.get("started"):
        with st.status("Preparing document store..."):
            st.write("Downloading data...")
            data_path = download_and_unzip(
                REPO_ZIP_URL, TARGET_EXTENSIONS, force_data_download
            )
            st.write("Initializing document store...")
            doc_store = DocStore(data_path, delete_persisted_db=delete_persisted_db)
            st.write("Persisting document store...")
            doc_store.persist()

            llm = OpenAI(temperature=temperature)
            retriever = doc_store.as_retriever(num_retrieved_docs)
            st.session_state["qa_chain"] = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=retriever,
                return_source_documents=True,
                return_generated_question=True,
            )
            st.session_state["started"] = True

    if not st.session_state.get("chat_history"):
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if question := st.chat_input("Enter a question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌")
            result = st.session_state["qa_chain"](
                {"question": question, "chat_history": st.session_state.chat_history}
            )
            message_placeholder.markdown(result["answer"])

        # Update chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.chat_history.extend([(question, result["answer"])])
        st.session_state.chat_history = st.session_state.chat_history[-4:]
