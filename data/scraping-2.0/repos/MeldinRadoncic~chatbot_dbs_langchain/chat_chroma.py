import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from streamlit_chat import message

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

st.write("Chroma Chatbot")

# Create vectorstore 
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory='db', embedding_function=embeddings)



def get_user_query():
    with st.form(key="my_form"):
        input_prompt = "Enter query"
        input = st.text_input(input_prompt, value="", key="input")
        submit_button = st.form_submit_button(label="Submit")
        if not input and submit_button:
            st.info("Input blank!")
    return input

def search_database(query):
    res_docs = db.similarity_search(query)
    chain = load_qa_chain(ChatOpenAI(temperature=0),chain_type="stuff", verbose=True)
    output = chain({'input_documents': res_docs, 'question': query})

    return output['output_text'] 

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def main():
    user_query = get_user_query()   
    if user_query:
        output = search_database(user_query)
        st.session_state['past'].append(user_query)
        st.session_state['generated'].append(output) 

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    main()
