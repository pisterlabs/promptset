from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import requests
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models.openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
import pickle

# Load environment variables
load_dotenv()

def extract_chunks_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    text_splitter = CharacterTextSplitter(
        chunk_size=1500,
        separator="\n",
        chunk_overlap=200)
    return text_splitter.split_text('\n'.join(line for line in lines if line))

def create_vector_store(chunks):
    print("Number of chunks: ", len(chunks))
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    #store = FAISS.from_texts(chunks, embeddings)

    #with open("faiss_store.pkl", "wb") as f:
    #    pickle.dump(store, f)

    store = pickle.load(open("faiss_store.pkl", "rb"))

    return store

def main():
    st.title("Chat with a web page")
    
    url = st.text_input('Enter the URL of your source text')
    
    if url is not None:
        chunks = extract_chunks_from(url)
        
        # Create the knowledge base object
        kb = create_vector_store(chunks) 
        
        query = st.text_input('Ask a question from your text')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = kb.similarity_search(query)
            print(docs)
  
            llm = ChatOpenAI(
                temperature=0.3,
                model_name="gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.write(response)
            
            
if __name__ == "__main__":
    main()
