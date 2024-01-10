import os
import streamlit as st
from PyPDF2 import PdfReader
import langchain
langchain.verbose = False
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import requests
from bs4 import BeautifulSoup
os.environ["OPENAI_API_KEY"] = "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"
def webscrap(name):
    # Replace this URL with the one you want to scrape
    url = f'https://www.{name}.com'

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
        return page_text
    else:
        return None
    
def main():
    print(os.getenv('OPENAI_API_KEY'))
    
    st.set_page_config(page_title="Webscrap chatbot")
    st.header("Webscrap chatbot")

    name = st.text_input("enter website name")
    
    web_data= webscrap(name)
    if web_data is not None:
        
        text = web_data
        # for page in pdf_reader.pages:
        #     text += page.extract_text()


        max_length = 1800
        original_string = text
        temp_string = ""
        strings_list = []

        for character in original_string:
            if len(temp_string) < max_length:
                temp_string += character
            else:
                strings_list.append(temp_string)
                temp_string = ""

        if temp_string:
            strings_list.append(temp_string)

        #split into chunks
        

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(strings_list, embedding=embeddings)

        user_question = st.text_input("Ask a question about your PDF")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()