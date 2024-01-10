import streamlit as st
# from dotenv import load_dotenv
from constants import openai_key
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"]=openai_key
def main():

    # load_dotenv()

    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")

    st.header("Hi, I am Chitti the Robot :robot_face:")
    
       
    # File Upload
    pdf = st.file_uploader("Upload your PDFs here and click on 'Process'")

    # Extract PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # Create Embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain({"input_documents": docs, "question": user_question},return_only_outputs=True)
            
            st.write(response['output_text'])



if __name__ == '__main__':
    main()