# your_script.py

import sys
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

def process_pdf(pdfText, input_text):
    # load_dotenv()

    text = pdfText

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Process the PDF content
    store_name = os.path.splitext(os.path.basename(pdf_path))[0]

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

    # Accept user questions/query
    query = input_text

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(response)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python my_script.py <pdf_path> <input_text>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    input_text = sys.argv[2]
    process_pdf(pdf_path, input_text)
