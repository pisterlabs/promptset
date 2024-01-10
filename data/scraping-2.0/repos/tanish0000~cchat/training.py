import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


load_dotenv()

traningfolder = 'traning/'
pdfs = []
text_files = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(f'{traningfolder}/{pdf}')
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_text_file(text_file):
  """Extracts the text from a text file.

  Args:
    text_file: The path to the text file.

  Returns:
    A string containing the text of the text file.
  """

  text = ""
  with open(text_file, "r") as f:
    for line in f:
      text += line
  return text

def text_chunks(text, text_files=[]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )

    chunks = text_splitter.split_text(text=text)

    for text_file in text_files:
        text = get_text_from_text_file(text_file)
        chunks += text_splitter.split_text(text=text)

    if not os.path.exists('traningdata.pkl'):
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"traningdata.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    return chunks

if not os.path.exists('traningdata.pkl'):
    folder = os.listdir(traningfolder)
    if len(folder) > 0:
        text = get_pdf_text(folder)
        _=text_chunks(text, text_files=text_files)

def main():

    if os.path.exists('traningdata.pkl'):
        with open(f"traningdata.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    st.header('chatpdf')

    inp = st.text_input('enter the query')
    if inp :
        docs = VectorStore.similarity_search(query=inp, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=inp)
        st.write(response)

main()
