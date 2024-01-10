import os.path
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #001EA7;
        color: white
    }
    [data-testid=stMarkdownContainer] h1, h2, .css-1offfwp a{
        color: white
    }
</style>
""", unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.title('LLM Document Chat App')
    st.markdown('''
        ## About
        This is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [langChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')

    add_vertical_space(5)
    st.write('Build for OEF 2023 hackday')


def main():
    st.header("Interact with your raw data")
    load_dotenv()
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # embeddings

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')

        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            with open(f"{store_name}.pk1", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings computation Completed')

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF files")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                res = chain.run(input_documents=docs, question=query)
                st.write(res)
                st.write(cb)


if __name__ == '__main__':
    main()

