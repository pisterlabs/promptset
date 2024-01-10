import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client 
import os


def get_vector_store():

    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    return vector_store


def main():
    # load env variables
    load_dotenv()
    
    # settings
    st.set_page_config(page_title='PDF-Chat', page_icon=':book:', layout='wide')

    st.header('Ask your Knowledge Base')
    
    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # create vectore store
    vector_store = get_vector_store()


    # create the chain 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    # accept user input
    if user_question := st.chat_input("Ask a question about your knowledge base:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
                with st.spinner('Thinking...'):
                    response = qa.run(user_question)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        

if __name__ == '__main__':
    main()