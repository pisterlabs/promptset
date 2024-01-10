import os

from langchain.llms import OpenAI
from langchain.prompts import (PromptTemplate, FewShotPromptTemplate)
from langchain.memory import ConversationBufferMemory 
from langchain.chains import (LLMChain, SimpleSequentialChain, SequentialChain)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate
import streamlit as st
import pandas as pd






st.title('PDF - Question Answering')
from PyPDF2 import PdfReader


# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# key_flag = True
key_flag = False
if os.path.exists('./openai_key.py'):
    print("Key file found")
    from openai_key import OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    key_flag = True
else:
    key = st.text_input("Enter the openai key")
    if key:
        os.environ['OPENAI_API_KEY'] = key
        key_flag = True


def train_and_save_doc_space(doc_path = './pdfs', embeddings = None, uploaded_file = False):
    myPdfReader = None

    pdf_files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.endswith('.pdf')]
    st.info(f"Number of pdf files found : {len(pdf_files)}")
    raw_text = ''
    
    if uploaded_file == False:
        for pdf_file in pdf_files:
            myPdfReader = PdfReader(pdf_file)
            for page in myPdfReader.pages:
                raw_text += page.extract_text()
    else:
        myPdfReader = PdfReader(doc_path)
        for page in myPdfReader.pages:
            raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=2000,
        chunk_overlap=200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    # show the length to the streamlit as log
    st.info(f"Length of the chunks: {len(texts)}")
    st.info("Creating the vector store")
    document_search_space = FAISS.from_texts(texts, embeddings)
    # print("Vector store created")
    st.info("Vector store created")

    document_search_space.save_local('faiss_doc_space')



def load_doc_space(name = 'faiss_doc_space', embeddings = None):
    if not embeddings:
        embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(name, embeddings)
    return db

if key_flag:
    # read all the pdf files from the dropbox directory

    # streamlit radio button
    # button = st.radio("Select the source of the pdfs to train", ('Local', 'Dropbox', 'Upload'), index=None)
    button = st.selectbox(
        'Select the source of the pdfs to train',
        ('Local', 'Upload', 'DropBox'),
        help = "Local source will used the pretrained doc. space",
        placeholder = "Select data source",
        index = None
    )   

    if button:
        st.write(f'You selected: :green[{button}] :sunglasses:')
        if button == 'Local':
            if not os.path.exists('./faiss_doc_space'):

                try:
                    train_and_save_doc_space(doc_path="./pdfs/", embeddings=OpenAIEmbeddings())
                except:
                    # st.error("Error in training the document search space")
                    pass

        elif button == 'Dropbox':
            pass

        elif button == 'Upload':
            uploaded_file = st.file_uploader("Upload a PDF file")
            if uploaded_file:
                try:
                    train_and_save_doc_space(doc_path = './pdfs/', embeddings = OpenAIEmbeddings(), uploaded_file = True)
                except:
                    # st.error("Error in uploading the file")
                    pass
            
            
        try:
            document_search_space = load_doc_space(embeddings=OpenAIEmbeddings())
        except:
            pass
            # st.error("Document search space not found")
            
        query = st.text_input("Enter the question")
        if query:
            docs = document_search_space.similarity_search(query,k = 5)
            df = pd.DataFrame([(doc.metadata['source'], doc.metadata['page'], doc.page_content) for doc in docs])
            uno_chain = load_qa_chain( llm=OpenAI(temperature=0.8), chain_type="stuff")
            st.write(uno_chain.run(input_documents = docs,question=query))
            with st.expander(f"Top 5 context for the question: {query}"):
                st.table(df)
