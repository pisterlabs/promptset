from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.header("ask your PDF")
    pdf = st.file_uploader("upload pdf", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        txt = ""
        for page in pdf_reader.pages:
            txt += page.extract_text()

        txt_spliter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = txt_spliter.split_text(txt)
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_qa = st.text_input("ask about your pdf")
        if user_qa:
            docs = knowledge_base.similarity_search(user_qa)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback():
                answer = chain.run(input_documents=docs, question=user_qa)
            st.write(answer)


if __name__ == '__main__':
    main()