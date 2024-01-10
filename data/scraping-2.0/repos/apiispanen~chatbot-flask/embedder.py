# STEPS TO GET OPENAI CONTEXTS:
# 1. Get the intent and entities from the query
# 2. Embed relevant documents
# 3. Get the top 5 most similar documents

from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from dotenv import load_dotenv

load_dotenv()
# from SpinnrAIWebService import apiconfig
# import apiconfig
import os
# import openai

# add OPENAI_API_KEY as an env

def get_vector_results(query, pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # print(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    # # embeddings
    store_name = pdf[:-4]
    # print(f'{store_name}')
    # st.write(chunks)
    try:
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            print('Embeddings Loaded from the Disk')
        else:
            print('Making new path')
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # query = input("Ask questions about your PDF file:")
        if query == '':
            return []
        else:
            docs = VectorStore.similarity_search(query=query, k=1)
    except Exception as e:
        print("*!*!*!*!ERRORRRR",e)
        return []
    # print the content of the docs
    return docs

# print(get_vector_results('what\'s tanvir\'s cgpa?', 'temp_pdf_file.pdf')[0].page_content)