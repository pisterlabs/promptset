# ref: https://www.youtube.com/watch?v=wUAUdEw5oxM&list=LL&index=1

from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    
    # print('OPENAI_API_KEY', oak)
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
    # load_dotenv()
    # print(os.getenv('OPENAI_API_KEY'))
    # exit()
    st.set_page_config(page_title="Ask PDF")
    st.header('Ask PDF')

    pdf = st.file_uploader("Upload a PDF ", type=["pdf"])
    if pdf:
        pdfreader = PdfReader(pdf)
        txt = ''
        for page in pdfreader.pages:
            txt += page.extract_text()
        
        # st.write(txt)

        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len
        )

        chunks = text_splitter.split_text(txt)   
        # st.write(chunks)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        question = st.text_input('Ask a question about the PDF')
        if question:
            docs = knowledge_base.similarity_search(question)
            # st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')
            with get_openai_callback() as cb:
                resp = chain.run(input_documents=docs, question=question)
                print(cb)

            st.write(resp)


if __name__ == '__main__':
    main()