from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from streamlit_chat import message
import asyncio
import glob
import os

DATA_STORE_DIR = "data_store"

if len(os.listdir(DATA_STORE_DIR)) == 1:
    print("The directory is empty. load the pdf files")
    files = glob.glob('documents/*.pdf')

    loaders = []
    for file_name in files:
        loaders.append(PyPDFLoader(file_name))

    pages =[]
    for loader in loaders:
        pages.extend(loader.load_and_split())

    vector_store = FAISS.from_documents(pages, OpenAIEmbeddings())
    vector_store.save_local(DATA_STORE_DIR)
    vector_store = FAISS.load_local(DATA_STORE_DIR, OpenAIEmbeddings())
else:
    print("The directory contains files. load the vector store")
    print(os.listdir(DATA_STORE_DIR))
    vector_store = FAISS.load_local(DATA_STORE_DIR, OpenAIEmbeddings())

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, request_timeout=120)  # Modify model_name if you have access to GPT-4
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever())

# Set the Streamlit page configuration, including the layout and page title/icon
st.set_page_config(layout="wide", page_icon="💬", page_title="ChatBot-PDF")

# Display the header for the application using HTML markdown
st.markdown(
    "<h1 style='text-align: center;'>FormSG-GPT, Chat with me on any FormSG matters you have! 💬</h1>",
    unsafe_allow_html=True)

async def main():
    try:
        st.session_state['ready'] = True

        # Define an asynchronous function for conducting conversational chat using Langchain
        async def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})

            # Add the user's query and the chatbot's response to the chat history
            st.session_state['history'].append((query, result['answer']))
            print(st.session_state['history'])
            return result['answer']

        # Set up sidebar with various options
        with st.sidebar.expander("🛠️ Settings", expanded=False):
            
            # Add a button to reset the chat history
            if st.button("Reset Chat"):
                st.session_state['reset_chat'] = True

        # If the chat history has not yet been initialized, do so now
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # If the chatbot is not yet ready to chat, set the "ready" flag to False
        if 'ready' not in st.session_state:
            st.session_state['ready'] = False
            
        # If the "reset_chat" flag has not been set, set it to False
        if 'reset_chat' not in st.session_state:
            st.session_state['reset_chat'] = False   
        
        # If the chatbot is ready to chat
        if st.session_state['ready']:

            # If the chat history has not yet been initialized, initialize it now
            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello ! Ask me anything 🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey ! 👋"]

            # Create a container for displaying the chat history
            response_container = st.container()
            
            # Create a container for the user's text input
            container = st.container()

            with container:
                        
                # Create a form for the user to enter their query
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Query:", placeholder="Ask your question here (:", key='input')
                    submit_button = st.form_submit_button(label='Send')
                    
                    # If the "reset_chat" flag has been set, reset the chat history and generated messages
                    if st.session_state['reset_chat']:
                        
                        st.session_state['history'] = []
                        st.session_state['past'] = ["Hey ! 👋"]
                        st.session_state['generated'] = ["Hello ! Ask me any HR matters 🤗"]
                        response_container.empty()
                        st.session_state['reset_chat'] = False

                # If the user has submitted a query
                if submit_button and user_input:
                    
                    # Generate a response using the Langchain ConversationalRetrievalChain
                    output = await conversational_chat(user_input)
                    print('output: ', output)
                    
                    # Add the user's input and the chatbot's output to the chat history
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            # If there are generated messages to display
            if st.session_state['generated']:
                
                # Display the chat history
                with response_container:
                    print('display chat history')
                    
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

    # Create an expander for the "About" section
    about = st.sidebar.expander("About 🤖")
    
    # Write information about the chatbot in the "About" section
    about.write("#### FormSG-GPT is an AI chatbot featuring conversational memory, designed to enable users to ask any formSG related queries in a more intuitive manner. 📄")
    about.write("#### Powered by [Langchain](https://github.com/hwchase17/langchain), [OpenAI](https://platform.openai.com/docs/models/gpt-3-5) and [Streamlit](https://github.com/streamlit/streamlit) ⚡")

#Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())