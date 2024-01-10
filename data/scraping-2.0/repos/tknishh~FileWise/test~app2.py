from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import textract
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
from PIL import Image

# Function to calculate similarity score between query and response
def calculate_similarity(query, response):
    query_embedding = llm.encode([query])[0]
    response_embedding = llm.encode([response])[0]
    similarity_score = cosine_similarity([query_embedding], [response_embedding])
    return similarity_score[0][0]

# Streamlit app
img = Image.open(r"title_image.png")
st.set_page_config(page_title="FileWise: Empowering Insights, Effortlessly.", page_icon= img)
st.header("Ask Your File📄")
file = st.file_uploader("Upload your file")

if file is not None:
    content = file.read()  # Read the file content once

    if file.type == 'application/pdf':
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == 'text/plain':
        text = content.decode('utf-8')
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = textract.process(content)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  

    chunks = text_splitter.split_text(text)
    # Load OpenAI embeddings and question-answering chain
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask your Question about the file")
    if query:
        docs = knowledge_base.similarity_search(query)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        # Evaluate relevance of the response
        similarity_score = calculate_similarity(query, response)
        st.text(f"Similarity Score: {similarity_score}")

        st.success(response)
