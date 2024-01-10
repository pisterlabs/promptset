from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import textract
from langchain.text_splitter import CharacterTextSplitter
from gensim.models import Word2Vec
import numpy as np
import faiss  # Import FAISS for similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
from PIL import Image

img = Image.open(r"title_image.png")
st.set_page_config(page_title="FileWise: Empowering Insights, Effortlessly.", page_icon=img)
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

    # Replace OpenAI embeddings with Word2Vec embeddings
    # Train a Word2Vec model on your text data
    word2vec_model = Word2Vec(chunks, vector_size=100, window=5, min_count=1, sg=0)

    # Convert the Word2Vec model to a dictionary of embeddings
    embeddings = {word: word2vec_model.wv[word] for word in word2vec_model.wv.index_to_key}

    # Convert the embeddings dictionary to a NumPy array
    embedding_matrix = np.array(list(embeddings.values()))

    # Create an index for similarity search using FAISS
    dim = word2vec_model.vector_size
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    query = st.text_input("Ask your Question about the file")
    if query:
        # Perform similarity search using FAISS
        query_vector = word2vec_model.wv[query]  # Get the vector for the query
        query_vector = query_vector.reshape(1, -1)  # Reshape for FAISS
        _, similar_doc_indices = index.search(query_vector, k=10)  # Find similar documents

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        # Get the similar documents based on indices
        similar_documents = [chunks[i] for i in similar_doc_indices[0]]

        # Combine similar documents into one text for question answering
        combined_text = "\n".join(similar_documents)

        response = chain.run(input_documents=[combined_text], question=query)
           
        st.success(response)
