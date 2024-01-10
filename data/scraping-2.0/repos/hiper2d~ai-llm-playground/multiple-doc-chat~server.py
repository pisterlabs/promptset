import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template


def get_text(docs):
    return "".join([doc.getvalue().decode('utf-8').replace("\n", " ") for doc in docs])


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="#",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        return

    with get_openai_callback() as cb:
        print(cb)
        resp = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = resp['chat_history']

        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, verbose=True)


def main():
    load_dotenv(find_dotenv())

    st.set_page_config(page_title="Chat with multiple docs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple docs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask me a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your docs")
        docs = st.file_uploader("Upload your docs and click on 'Process'", accept_multiple_files=True, type=["txt"])
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_text(docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
