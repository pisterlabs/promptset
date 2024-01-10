import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
from PIL import Image
import os
import pandas as pd
from PIL import Image
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
from pathlib import Path
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import pickle
from streamlit_chat import message


# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"


def func(filename):
    if(filename!=None):
        print(filename)
        reader = PdfReader(filename)
        
        # printing number of pages in pdf file
        pdf_len = len(reader.pages)
        
        # getting a specific page from the pdf file
        final_text=''

        final_list=list()

        for i in range(pdf_len):
                page = reader.pages[i]
                text = page.extract_text()
                final = text.replace("\n"," ")
                final_text=final_text+text

                final_list.append(final)        
        
        # extracting text from page

        new_list = list(filter(lambda x: x != '', final_list))
        # print(new_list)
        # print(len(new_list))
        return new_list
    
    
def newList(filename):
    new_list=func(filename)
    embeddings = OpenAIEmbeddings()

    return new_list,embeddings




def chatur_gpt(filename):
    new_list,embeddings= newList(filename)
    if(new_list!=None):

        if(len(new_list)!=0):

            docsearch = FAISS.from_texts(new_list, embeddings)
            qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
            qa = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=docsearch)
        
    return qa
    
    

def generate_response(prompt,qa):
    message = qa.run(prompt)
    return message

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

def main():
    st.title("Custom GPT")
    st.write("Upload a file to train using GPT-3")
    file = st.file_uploader("Upload a file", type=["pdf"])
    
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    if file is not None:
        if os.path.isfile(file.name) == False:
            save_folder = os.getcwd()
            save_path = Path(save_folder, file.name)
            with open(save_path, mode='wb') as w:
                w.write(file.getbuffer())
        #st.write(file.read())
        new_list,embeddings= newList(file.name)
        if(new_list!=None):

            if(len(new_list)!=0):
                docsearch = FAISS.from_texts(new_list, embeddings)
                qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
                qa = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=docsearch)
#             st.write(file.name)
#             res = chatur_gpt
                user_input = get_text()

                if user_input:
                    output = generate_response(user_input,qa)
                    # store the output 
                    st.session_state.past.append(user_input)
                    st.session_state.generated.append(output)

                if st.session_state['generated']:
                    for i in range(len(st.session_state['generated'])-1, -1, -1):
                        message(st.session_state["generated"][i], key=str(i))
                        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

main()