from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv 
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import streamlit as st
load_dotenv()

def main():
    st.set_page_config(page_title="👨‍💻 Talk with your CSV")
    st.title("👨‍💻 Talk with your Website")
    st.write("Please insert your link.")
    url = st.text_input("Insert The link")

    query = st.text_input("Send a Message")
    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        # Load data from the specified URL
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split the loaded data
        text_splitter = CharacterTextSplitter(separator='\n', 
                                        chunk_size=1000, 
                                        chunk_overlap=200)

        docs = text_splitter.split_documents(data)

        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings()

        # Create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=docs, 
                                        embedding=openai_embeddings,
                                        persist_directory=DB_DIR)

        vectordb.persist()

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-4')

        # Create a RetrievalQA from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the query and return the result
        result = qa(query)
        st.write(result)

if __name__ == '__main__':
    main()

# def retrieve_and_query(url: str, query: str):
#     # Setup paths
#     ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
#     DB_DIR: str = os.path.join(ABS_PATH, "db")

#     # Load data from the specified URL
#     loader = WebBaseLoader(url)
#     data = loader.load()

#     # Split the loaded data
#     text_splitter = CharacterTextSplitter(separator='\n', 
#                                       chunk_size=1000, 
#                                       chunk_overlap=200)

#     docs = text_splitter.split_documents(data)

#     # Create OpenAI embeddings
#     openai_embeddings = OpenAIEmbeddings()

#     # Create a Chroma vector database from the documents
#     vectordb = Chroma.from_documents(documents=docs, 
#                                      embedding=openai_embeddings,
#                                      persist_directory=DB_DIR)

#     vectordb.persist()

#     # Create a retriever from the Chroma vector database
#     retriever = vectordb.as_retriever(search_kwargs={"k": 3})

#     # Use a ChatOpenAI model
#     llm = ChatOpenAI(model_name='gpt-4')

#     # Create a RetrievalQA from the model and retriever
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#     # Run the query and return the result
#     result = qa(query)
#     return result


