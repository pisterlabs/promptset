from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
from PyPDF2 import PdfReader
import pickle
import os
from trulens_eval import Feedback, TruChain, feedback, Select
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
ENDPOINT = os.getenv('OPENAI_API_BASE')
OPENAI_VERSION = os.getenv('OPENAI_API_VERSION')

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown("""
    ## About
    This app is an LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
    """)

def main():
    st.header('Document Skimming Assistant')
    st.write('Upload a PDF file and ask questions about its content.')

    pdf = st.file_uploader('Choose a PDF file: ', type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
        chunks = text_splitter.split_text(text=text)
    
        # Embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            st.write('Embeddings loaded from the disk')
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, openai_api_base=ENDPOINT,
                                          openai_api_type='azure', deployment='bot-embedding', chunk_size=1, max_retries=20)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, file=f)
        
        # Chat with users
        query = st.text_input('Hi! How can I support you?') 
        if query:
            docs = vectorstore.similarity_search(query=query, k=3)
            st.write(docs)
            azure_chat_llm = AzureChatOpenAI(deployment_name='bot-chat', model='gpt-35-turbo-16k', temperature=0.1)
            qa_source_chain = RetrievalQA.from_chain_type(llm=azure_chat_llm, chain_type='stuff', retriever=vectorstore.as_retriever(),
                                                          return_source_documents=True)
            azure_openai = feedback.AzureOpenAI(deployment_id='bot-chat', model_engine="gpt-35-turbo-16k")
            grounded = feedback.Groundedness(groundedness_provider=azure_openai)
            grounded.summarize_provider = azure_openai
            f_groundedness = Feedback(grounded.groundedness_measure).on(Select.Record.calls[2].rets.context).on_output()
            f_relevance = Feedback(azure_openai.relevance).on_input_output()
            truchain = TruChain(qa_source_chain, app_id='chat-bot', feedbacks=[f_groundedness, f_relevance], tags = "prototype")
            response = truchain.call_with_record(inputs={"query": query})
            st.write(response[0]['result'])
main()