import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys

DB_FAISS_PATH = "vectorstore/db_faiss"




# loading model
def load_llm():
    # Load locally downloaded nodel
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.1
    )
    return llm


# st.title("Chat with WhistleOut Data")
st.markdown("<h3 style='text-align: center; color: grey'>Chat With WhistleOut</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey'>UTS MDSI</h3>", unsafe_allow_html=True)

# load the CSV data
loader = CSVLoader(file_path="data/DimUTSProduct.csv", encoding="utf-8", csv_args={
    'delimiter': ','
})
data = loader.load()
st.json(data)
# Split the texts into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)

# call the llm
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hey! ask me anything about your data!"]
if 'past' not in st.session_state:
    st.session_state['past'] = ["hey!"]

# container for the chat history
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query: ", placeholder="Let's answer your question", key='input')
        submit_button = st.form_submit_button(label="chat")

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")