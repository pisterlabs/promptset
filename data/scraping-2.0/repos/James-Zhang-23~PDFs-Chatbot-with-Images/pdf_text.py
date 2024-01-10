import fitz
import os
import shutil
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Iterating over the pages in the PDF
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def generate_embedding(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    # check if the vector store exists
    if os.path.exists("faiss_index"):
        # load the vector store
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        # embedding the text chunks and add to current vector store
        vectorstore.add_texts(texts=text_chunks, embedding=embeddings)
    else:
        # create a new vector store
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

if __name__ == "__main__":
    # clear the faiss_index folder
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    pdf_docs = []
    # Open all the PDF file in the folder path processed_pdfs
    for file_name in os.listdir("processed_pdfs"):
        pdf_docs.append(fitz.open("processed_pdfs/" + file_name))

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = generate_embedding(text_chunks)

    # Save the vectorstore to the specified path
    vectorstore.save_local("faiss_index")

    
