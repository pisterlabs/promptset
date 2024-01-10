from dotenv import load_dotenv
import os
import streamlit as st
import langchain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks import StdOutCallbackHandler
langchain.debug = True
langchain.verbose=True

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_DIR = os.path.join(os.path.dirname(__file__), 'db')

def instantiate():
    """
    Instantiate objects

    Args:
        None

    Returns:
        chat_model: the chat model
        tools: the tools
        search: the search
        loader: the loader
        text_splitter: the text splitter
    """

    handler = StdOutCallbackHandler()
    llm = ChatOpenAI(model_name='gpt-4', verbose=True)
    openai_embeddings = OpenAIEmbeddings()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                       chunk_overlap=40)
    
    return handler, llm, openai_embeddings, text_splitter


def main():
    """
    Run the main function

    Args:
        None

    Returns:
        None
    """

    handler, llm, openai_embeddings, text_splitter = instantiate()

    st.title("🦜🔗 AI 42 Vienna")
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')
    url = st.text_input("Insert The website URL")
    prompt = st.text_input("Ask a question (query/prompt)")

    if st.button("Submit Query", type="primary"):

        loader = WebBaseLoader(url)

        data = loader.load()
        
        docs = text_splitter.split_documents(data)
        
        #vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=DB_DIR)
        
        vectordb = Chroma(persist_directory="ai_42_vienna/llm/db", embedding_function=openai_embeddings)
        
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), 
                                                          llm=llm)
        qa = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever_from_llm,
                                         return_source_documents=True,
                                         callbacks=[handler])
        
        result = qa(prompt, callbacks=[handler])

        st.write("Answer: ")
        st.write(result["result"])
        st.write("Source Documents: ")
        st.write(result["source_documents"])

if __name__ == '__main__':
    main()


#https://www.42vienna.com/
#how to apply?