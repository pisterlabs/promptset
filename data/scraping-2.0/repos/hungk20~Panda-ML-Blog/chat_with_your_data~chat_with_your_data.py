import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from streamlit_chat import message as st_message

# load db that store new data
DATA_STORE_DIR = "data_store"
vectordb = FAISS.load_local(DATA_STORE_DIR, embeddings=OpenAIEmbeddings())
# initialize language model and question & answer retrieval from langchain
llm = ChatOpenAI(temperature=0.0)
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(), verbose=True
)


def generate_answer():
    # get user message and add to chat history
    user_message = st.session_state.input_text
    st.session_state.history.append({"message": user_message, "is_user": True})
    # get bot message and add to chat history
    bot_message = qa_stuff.run(user_message)
    st.session_state.history.append({"message": bot_message, "is_user": False})
    # clear input text
    st.session_state["input_text"] = ""


# initialize
if "history" not in st.session_state:
    st.session_state.history = []

# define chatbot interface
st.title("ChatGPT + dữ liệu mới")
st.text_input("Chat với tôi", key="input_text", on_change=generate_answer)

# display chat message
for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i))
