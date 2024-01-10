from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
import cohere
import langchain
from langchain.chains.question_answering import load_qa_chain


with st.sidebar:
    st.title('PDF CHATBOT📄🤖')
    st.markdown('''
    ## About
    This app is an Cohere-powered chatbot for PDFs. 
    ''')
    add_vertical_space(5)
    st.write('Made by Abhinav Lodha')
    
def main():
    langchain.verbose = False
    #UI
    st.header("PDF CHATBOT📄🤖")
    st.write("Chat with your PDFs!")
    pdf =st.file_uploader("Upload a PDF file", type="pdf")
    
    # if the pdf is uploaded
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        
        # extracting the text from the pdf
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # creating the desired text splitter
        text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=  1000,
           chunk_overlap= 150,
           length_function=len                                     
        )
        
        # creating chunks of the text so that it can be used by the model
        chunks = text_splitter.split_text(text=text)
        
        embeddings = CohereEmbeddings(cohere_api_key="0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd")
        VectorStore=FAISS.from_texts(chunks, embedding=embeddings)
        
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            client = cohere.Client("0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd")
            llm = Cohere(client=client,cohere_api_key="0ngkfNjlXOvHsR4PzcOCQd6vaRycOV4BkSkNVAKd")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
    
if __name__ == '__main__':
    main()