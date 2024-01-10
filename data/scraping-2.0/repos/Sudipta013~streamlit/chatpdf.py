from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
import os

headers = {
    "authorization": st.secrets["auth_token"],
   "content-type": "application/json"
}
open_ai_key = st.secrets["auth_token"]
huggingface_key = st.secrets["huggingface_key"]


#using 1 pdf 
def main():
    st.set_page_config(page_title="chatPdf", page_icon="pdf")

    #CSS
    st.markdown("<h1 style='text-align: center; font-family:Abril Fatface ; -webkit-text-stroke: 1px black ;font-size: 70px; padding-bottom: 15px; color: rgb(255, 255, 255) ;'>Ask Your PDF</h1>", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;font-family:Nunito ;font-size:18px;color: rgba(255,255,255,0.5); padding-top: 15px'>
                Your PDF AI - like ChatGPT but for PDFs. Summarize and answer questions for free.
                AskPDF can be really helpful in situations when you have to sift through a large amount of text. 
                It can help you find a particular topic, summarize a PDF, or understand the topic in simpler terms and you can also ask follow-up questions about it.
                </h5>""",unsafe_allow_html=True)
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    bg = """
        <style> [data-testid="stAppViewContainer"]{
        background: rgb(6,36,39);
        }
        </style>
        """
    st.markdown(bg, unsafe_allow_html=True)

    # Add the yellow bottom bar
    bottom_bar_html = """
    <style>
    .bottom-bar {
        background-color: #FFA500;
        padding: 5px;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-family: 'Russo One';
        font-size: 20px;
    }
    </style>
    <div class="bottom-bar">
        <span style="color: white; font-weight: bold;">The Techie Indians</span>
    </div>
    """
    st.markdown(bottom_bar_html, unsafe_allow_html=True)

    # upload file
    pdf = st.file_uploader("")
        
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        st.subheader("Choose your embedding model")
        embedding_option  = st.radio(
            "Choose Model", ["OpenAI", "HuggingFace"])

        # create embeddings
        if embedding_option == "OpenAI":
            embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)
        elif embedding_option == "HuggingFace":
            embeddings = HuggingFaceEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        st.subheader("Chat...")
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            if embedding_option == "OpenAI":
                llm = OpenAI(openai_api_key=open_ai_key)
            elif embedding_option == "HuggingFace":
                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token=huggingface_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
