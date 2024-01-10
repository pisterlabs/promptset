import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from PyPDF2 import PdfReader


def get_embeddings():
    return HuggingFaceEmbeddings()

def get_qa_chain():
    return load_qa_chain(OpenAI(), chain_type="stuff")

def extract_pdf(file):
    """ Takes User name and goes to that folder in DB and extracts Text from all the PDFs """
    raw_text = ""
    for pdf_file in os.listdir(file):
        doc_reader=PdfReader(f"{file}/{pdf_file}")
        for page in doc_reader.pages:
            raw_text += " " + page.extract_text()
    return raw_text


embd_folder_1,embd_folder_2="parivartan_stored_embd","parivartan_gen_nxt_stored_embd"
Path(embd_folder_1).mkdir(parents=True, exist_ok=True)
Path(embd_folder_2).mkdir(parents=True, exist_ok=True)

embeddings = HuggingFaceEmbeddings()
        
        
def scrape_and_create_embeddings_1(url):
    complete_text=""
    try:
        response = requests.get(url)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, 'html.parser')   ## Parse HTML content 
        all_text = soup.get_text(separator=' ')                 ##  text extraction from html tags

        cleaned_text = "\n".join(line.strip() for line in all_text.splitlines() if line.strip())   # join line and remove empty line
        with open("Parivartan.txt", 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

        pdf_text=extract_pdf("parivartan_stored_pdfs")
        complete_text=cleaned_text+pdf_text


        print("****** Website Scrape done and saved")
        st.info("'parivartan' Website Scrape done for and saved")

        try:
            convert_embeddings(complete_text,embd_folder_1)
            
            print("****** Embd conversion done")
            st.info(" 'parivartan' Embeddings created")

        except Exception as e:
            print("**** Issue in Converting embd",e)
     
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")



def scrape_and_create_embeddings_2(url):
    complete_text_2=""
    try:
        response = requests.get(url)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, 'html.parser')   ## Parse HTML content 
        all_text = soup.get_text(separator=' ')                 ##  text extraction from html tags

        cleaned_text_1 = "\n".join(line.strip() for line in all_text.splitlines() if line.strip())   # join line and remove empty line
        with open("Parivartan_gen_nxt.txt", 'w', encoding='utf-8') as file:
            file.write(cleaned_text_1)

        pdf_text=extract_pdf("parivartan_gen_nxt_stored_pdfs")
        complete_text_2=cleaned_text_1+pdf_text


        print("****** Website Scrape done and saved")
        st.info("'parivartan gen nxt' Website Scrape done for and saved")

        try:
            convert_embeddings(complete_text_2,embd_folder_2)
            
            print("****** Embd conversion done")
            st.info(" 'parivartan gen nxt' Embeddings created")

        except Exception as e:
            print("**** Issue in Converting embd",e)
     
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")



# Example usage
#url = "https://en.wikipedia.org/wiki/Main_Page"
#output_file = "output.txt"
#scrape_and_create_embeddings(url, output_file)
def convert_embeddings(text,saved_folder):
    embeddings = HuggingFaceEmbeddings()
    splitter = CharacterTextSplitter(separator=".", chunk_size=200, chunk_overlap=100, length_function=len)
    chunk_lst = splitter.split_text(text)
    # Convert chunks to embeddings
    FAISS_db = FAISS.from_texts(chunk_lst,embeddings)
    FAISS_db.save_local(saved_folder)




def load_and_answer_questions(question,embd_folder):
    api="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"#"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"
    os.environ["OPENAI_API_KEY"]=api

    FAISS_db = FAISS.load_local(embd_folder,HuggingFaceEmbeddings())
    chain = get_qa_chain()    
    docs = FAISS_db.similarity_search(question)
    print(f"Question: {question}")
    answer = chain.run(input_documents=docs, question=question)

    
    print(f"Answer: {answer}")
    return answer


#question="india tunnel collapse?"    
#load_and_answer_questions(question)



