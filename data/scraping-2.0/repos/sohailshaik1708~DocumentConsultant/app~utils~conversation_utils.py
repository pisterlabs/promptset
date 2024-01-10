import streamlit as st
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from app.html.htmlTemplates import bot_template, user_template


def get_conversation_chain(vectorstore):
    """
    Creates a conversational retrieval chain.

    Args:
        vectorstore (langchain.vectorstores.FAISS): Vector store for text chunks.

    Returns:
        langchain.chains.ConversationalRetrievalChain: Conversational retrieval chain.
    """
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(input):
    """
    Handles user input and generates bot response.

    Args:
        input (str): User input.

    Returns:
        None
    """
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.write("Please upload your PDFs and click on 'Process' to start the conversation.")
        return
    response = st.session_state.conversation({'question': input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



