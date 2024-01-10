import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv
import uuid
from streamlit_chat import message

# load env vars
load_dotenv()

# OpenAI embeddings
embeddings = OpenAIEmbeddings(client=None)

# load FAISS db
db = FAISS.load_local("faiss_db", embeddings)

# memory for chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# llm
llm = OpenAI(temperature=0, client=None)

# init retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory)

# Streamlit app
st.title("🤖 Ask me about blog.baeke.info")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(question):
	result = qa({"question": question})
	return result["answer"]

def get_text():
    input_text = st.text_input(key="input", label="Type a question and press Enter")
    return input_text 

user_input = get_text()

if user_input:
    output = query(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
        