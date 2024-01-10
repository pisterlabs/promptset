
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pickle

import streamlit as st

from langchain.globals import set_debug
set_debug(True)

st.set_page_config(page_title="Ask PDX")
st.title("Ask questions about Portland's City Charter, Code, and Policies")

# create a function which returns a retriever using FAISS
def faiss_retriever(): 
    with open("faiss_store.pkl", "rb") as f:
      store = pickle.load(f)
    retriever = store.as_retriever()
    return retriever

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=st.session_state.openai_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    response = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    sources = list(set([doc.metadata['source'] for doc in response['source_documents']]))    
    result = (response['answer'], sources)
    st.session_state.messages.append((query, result[0]))
    return result

def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")


def boot():
    input_fields()

    if not st.session_state.openai_api_key:
        st.warning(f"Please provide the missing fields.")
        
    st.session_state.retriever = faiss_retriever()

    #
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        result = query_llm(st.session_state.retriever, query)
        message = f"{result[0]}\n\n*Sources:* {result[1]}"
        st.chat_message("ai").write(message)

if __name__ == '__main__':
    boot()