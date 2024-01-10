import os
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from pinecone_embeddings import embeddings, index_name
"""
    Embeddings are vector representations of data, usually high-dimensional, 
    distilled into a lower-dimensional form while retaining essential features. 
    In NLP, word or sentence embeddings capture semantic meanings and are commonly 
    used in tasks such as text classification, clustering, and information retrieval. 
    An embeddings database like Pinecone or FAISS is designed to quickly look up and 
    perform nearest neighbor search on these embeddings.

    The different embeddings we have available are just a reflection of the providers
    different algorithms KNN search, FAISS, etc.  We can use any of these to generate
    our embeddings and store them in our database.
"""
"""
    Sometimes, the full documents can be too big to want to retrieve them as is. 
    In that case, what we really want to do is to first split the raw documents 
    into larger chunks, and then split it into smaller chunks. We then index the 
    smaller chunks, but on retrieval we retrieve the larger chunks (but still not 
    the full documents).
"""

# Streamlit app
st.subheader('Generative Q&A with LangChain & Pinecone')
            
query = st.text_input("Enter your query")

if st.button("Submit"):
    # Validate inputs
    if not os.environ["OPENAI_API_KEY"] or not os.environ["PINECONE_API_KEY"] or not os.environ["PINECONE_ENV"]:
        st.warning(f"Please upload the document and provide the missing fields.")
    else:
        try:
            vectordb = Pinecone.from_existing_index(index_name, embeddings)
            retriever = vectordb.as_retriever()

            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
            qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
            response = qa.run(query)
            
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
