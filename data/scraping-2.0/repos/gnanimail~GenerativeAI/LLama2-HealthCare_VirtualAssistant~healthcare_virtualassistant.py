import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory

# load the mental health document
document_mh = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents_ = document_mh.load()

# split the document into chunks
text_splitter_mh = RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=50)
text_chunks_mh = text_splitter_mh.split_documents(documents_)


# create embeddings
embeddings_mh = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                         model_kwargs={"device":"cpu"})


# create vector store
knowledge_base = FAISS.from_documents(text_chunks_mh, embeddings_mh)


# load the foundation model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config={"max_new_tokens":128,
                            "temperature":0.2})

memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                              retriever=knowledge_base.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

############################################################################################################
st.title("HealthCare ChatBot 🧑🏽‍⚕️")

def virtualassistant(query):
    response_ = chain({"Question":query, "chat_history":st.session_state["history"]})
    st.session_state["history"].append(query, response_["answer"])
    return response_["answer"]

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"]=[]

    if "generated" not in st.session_state:
        st.session_state["generated"]=["Hello! Ask me anything about 🤗"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = virtualassistant(user_input)

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()








