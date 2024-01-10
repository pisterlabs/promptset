import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PIL import Image
import pytesseract
from transformers import pipeline
from langchain.document_loaders import UnstructuredWordDocumentLoader
import docx2txt
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)



st.set_page_config(page_title='Doc-Q&A ',page_icon="./logo.png",layout ="wide") 
image = "./logo.png"
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.header("Chat with Documents 💬")

from streamlit_extras.app_logo import add_logo
add_logo("logo.png", height=60)

# upload a PDF file
uploaded_files = st.file_uploader("Upload your Document(pdf,text,docx,images)",accept_multiple_files=True)
#st.write(uploaded_files)
name = []
for i in uploaded_files:
     name.append(i.name)
st.write(name)
model_path = st.selectbox('Select an Option',["google/flan-t5-base","MBZUAI/LaMini-Flan-T5-783M","google/flan-t5-xxl","impira/layoutlm-document-qa",'bigscience/bloom-560m',"google/flan-t5-xl"])
new_list = []
data_str_final = ""

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
}
def load_single_document(uploaded_files):
    # Loads a single document from a file path
    file_extension = os.path.splitext(uploaded_files)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(uploaded_files)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]

def load_documents(uploaded_files):
    # Loads all documents from the source documents directory
    all_files = uploaded_files
    
    docs = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(uploaded_files, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            docs.append(load_single_document(source_file_path))
    return docs

documents = load_documents(uploaded_files)
if documents[0].name.endswith('.png') or documents[0].name.endswith('.jpg') or documents[0].name.endswith('.jpeg'):
            #   s = document.getvalue().decode()
      pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Anubhav194010\Tesseract-OCR\tesseract.exe"
      img = Image.open(documents)
      st.image(img)
      loader = UnstructuredImageLoader(img)
      data = loader.load()
      #st.write(data)
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
      chunks = text_splitter.split_documents(data)


else :
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)

      chunks = text_splitter.split_documents(documents)

      embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

      # # load the vectorstore
      db = FAISS.from_documents(chunks, embeddings)
      #   st.write("Embeddings Done")

if model_path.endswith("-qa"):
      nlp = pipeline("document-question-answering",model=model_path)
      query = st.text_input("Ask questions from your documents:")
      result = nlp(img,query)
      st.write(result)

else:

      #st.write(type(model_path))
      llm = HuggingFaceHub(repo_id=model_path, model_kwargs={"temperature":0.1})
      st.write("Model Loaded - ",model_path)
      # Accept user questions/query
      query = st.text_input("Ask questions from your documents:")
      if query:
           
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3}), return_source_documents=False, verbose=False)
            result = qa({"query": query})
            st.write(result)




                        #st.write(chunks)
                  # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
                  # pd_lst = []
                  # for i in new_list:
                  #     chunks = text_splitter.create_documents(i)
                  #     pd_lst.append(chunks)
            


                        
      
                  
                        # Text Splitter
                  # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 10)
                  # chunks = text_splitter.split_documents(data)
           

          
                  # # # embeddings
                  #   embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                  # # load the vectorstore
                  #   db = FAISS.from_documents(chunks, embeddings)
                  #   st.write("Embeddings Done")



                  #   