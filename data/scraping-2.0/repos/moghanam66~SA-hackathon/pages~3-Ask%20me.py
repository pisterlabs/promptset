import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from utils import *
    
def main():

    # Congfiger the page attributes
    st.set_page_config(
        page_title='Saudi Tourism',
        page_icon=":star:",layout="wide")
    # Set the background
    set_background('wallpaper.jpeg')
    add_logo("logo4.png")
    # laod the scrapped data of FAQ
    tmp_file_path='FAQ.csv'
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
    
    # convert them to embeddings and store them in a vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(data, embeddings)

    # Represent two sentence to represente how the chat works
    if 'history' not in st.session_state:
            st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask any thing about tourism in KSA " +  " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            # get the user input
            user_input = st.text_input("Question:", placeholder=" ", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            # get the output from open ai based on our data stored in the vector store
            output = conversational_chat(user_input,vectorstore)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
                
                
    # Configure how the chat would look like    
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs") 

# Cache the answer to optimize the performance and reduce any wasted cost. 
@st.cache_data
def conversational_chat(query,_vectorstore):
        # config the chain with the llm and the vector store
        chain = ConversationalRetrievalChain.from_llm(
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.0,model_name='gpt-3.5-turbo'),
            retriever=_vectorstore.as_retriever())
        
        result = chain({"question": query, 
        "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]

if __name__ == "__main__":
    main()
      
