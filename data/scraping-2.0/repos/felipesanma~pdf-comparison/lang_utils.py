import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader


def get_text_splitter(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    # st.info(f"Extracting text from PDF {pdf_file.name}")
    for page in pdf_reader.pages:
        text += page.extract_text()
    # st.info(f"Getting Chunks from {pdf_file.name}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, length_function=len
    )
    result = text_splitter.split_text(text)
    return result


def create_qa_retrievals(pdf_file_list: list, OPENAI_API_KEY):

    qa_retrievals = []
    for pdf in pdf_file_list:
        # st.info(f"Processing {pdf.name}")
        texts = get_text_splitter(pdf)
        # st.info(f"Converting PDF {pdf.name} to embedding")
        docsearch = Chroma.from_texts(
            texts,
            OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            metadatas=[{"source": f"{i}-{pdf.name}"} for i in range(len(texts))],
        )
        st.info(f"Saving {pdf.name} to vector DB")
        qa_tmp = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=docsearch.as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            ),
            return_source_documents=True,
        )
        qa_retrievals.append(qa_tmp)

    return qa_retrievals


def ask_to_all_pdfs_sources(query: str, qa_retrievals):
    responses = []
    progress_text = f"Asking '{query}' to all PDF's"
    total_retrievals = len(qa_retrievals)
    my_bar = st.progress(0, text=progress_text)
    for count, qa in enumerate(qa_retrievals):
        result = qa({"query": query})
        tmp_obj = {
            "query": query,
            "response": result["result"],
            "source_document": result["source_documents"][0]
            .metadata["source"]
            .split("-")[1],
        }
        responses.append(tmp_obj)
        percent_complete = (count + 1) * 100 / total_retrievals
        my_bar.progress(int(percent_complete), text=progress_text)

    return responses
