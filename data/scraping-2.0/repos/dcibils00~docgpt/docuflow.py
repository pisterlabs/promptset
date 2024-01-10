# Imports
from io import StringIO
import os 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredURLLoader

from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
from tempfile import NamedTemporaryFile
import tempfile

###
### run with $ streamlit run docuflow.py
###

# Set API keys and the models to use
API_KEY = ""
#model_id = "gpt-3.5-turbo"
model_id = "gpt-4"
gpt_temperature = 0.4

# Add your openai api key for use
os.environ["OPENAI_API_KEY"] = API_KEY

#loaders = PyPDFLoader()

# Setup streamlit app
st.set_page_config(page_title="DocGPT", page_icon="🚀", layout="wide")


# Display the page title and the text box for the user to ask the question

st.title('🚀 DocGPT / Chat with your documents')
st.text('Uses OpenAI GPT (C) Diego Cibils, 2023')
st.text("Using OpenAI model: " + model_id)
        
# Display the file uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf','txt'])
#url_input = st.text_input("Or enter a URL to scrape")
#url_button = st.button("Load URL", key="load_url", help="Click to load the URL")
#if url_button:
#    url = url_input.strip()
#    st.write("Loading URL: " + url)

#    loaders = UnstructuredURLLoader(urls = [url])
#    loaders.load()
#    print(loaders)
#    index = VectorstoreIndexCreator().from_loaders([loaders])
#    st.write("URL processed successfully!")

if uploaded_file is not None:
    print("Using File: " + uploaded_file.name)

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.read())
        
        if (uploaded_file.name.endswith('.txt')):
            loaders = TextLoader(tf.name)  
        elif (uploaded_file.name.endswith('.pdf')):
            loaders = PyPDFLoader(tf.name)
        
        index = VectorstoreIndexCreator().from_loaders([loaders])

    st.write("File processed successfully!")

prompt = st.text_input("Enter your question here (in any language):")

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM
    if index:
        response = index.query(llm=OpenAI(model_name = model_id, temperature=gpt_temperature), 
                            question = prompt, 
                            chain_type = 'stuff')

        #------------------------------------------------------------------
        # Save history

        # This is used to save chat history and display on the screen
        if 'answer' not in st.session_state:
            st.session_state['answer'] = []

        if 'question' not in st.session_state:
            st.session_state['question'] = []    

        # Add the question and the answer to display chat history in a list
        # Latest answer appears at the top
        st.session_state.question.insert(0,prompt  )
        st.session_state.answer.insert(0,response  )
        
        # Display the chat history
        k = 0
        for i in range(len( st.session_state.question)) :
            k+=1
            message(st.session_state['question'][i], is_user=True, key=k)
            k+=1
            message(st.session_state['answer'][i], is_user=False, key=k)
