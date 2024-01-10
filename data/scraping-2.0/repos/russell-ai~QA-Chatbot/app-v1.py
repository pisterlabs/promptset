import streamlit as st
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64


#load api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.title('RUSSELL BOT 🤖')
    st.markdown('''
    ## About APP:
    - This app is a demo for pdf based chatbot.
    - You can upload your pdf file and ask questions about it.


    ## About me:
    - [Linkedin](https://www.linkedin.com/in/rcaliskan/)
    - [Github](https://github.com/russell-ai)
    
    ''')

    add_vertical_space(4)
    # write sth about copy right
    st.write('Made with ❤️ by Russell AI')
    st.write('© 2023 Russell AI. All rights reserved.')

def main():
    st.header("Chat with your PDF file 📄")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        st.write(pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        # create a folder for store pdf name
        # get pdf name without .pdf
        db_folder = "db"
        store_name = "".join(pdf.name.split(".")[:-1]).strip()
        if not os.path.exists(db_folder):
            os.mkdir(db_folder)
        # check if file exists
        vectorstore_path = os.path.join(db_folder,store_name+".pkl")

        
        if os.path.exists(vectorstore_path):
            with open(vectorstore_path,"rb") as f:
                vectorstore = pickle.load(f)
            st.write("File already exists. Embeddings loaded from the your folder (disks)")
        else:
            # embedding method (Openai)
            embeddings = OpenAIEmbeddings()

            # create a vector store and store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(vectorstore_path,"wb") as f:
                pickle.dump(vectorstore,f)
            
            st.write("Embedding created and stored in the folder")
        
        # User questions/query

        query = st.text_input("Ask questions about related your upload pdf file")
        st.write(query)
        cancel_button = st.button('Cancel');
        if cancel_button:st.stop()
        

        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            
            # openai llm process  
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)

if __name__=="__main__":
    main()  
