import os
import streamlit as st
from langchain import OpenAI
from haystack.document_store import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.pipeline import DocumentSearchPipeline

# Retrieve the OpenAI API key from an environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create the document store
document_store = FAISSDocumentStore()

# Load documents into the document store
# ... (code to load your documents into the document store)

# Create the retriever
retriever = DensePassageRetriever(document_store=document_store)

# Create the pipeline
pipeline = DocumentSearchPipeline(retriever=retriever)

def run_streamlit_app():
    st.title("Question Answering System")

    # Display the input form to enter the question
    query_str = st.text_input("Enter your question")

    if st.button("Submit"):
        # Query the pipeline and get the response
        prediction = pipeline.run(query_str)

        # Extract the answer from the prediction
        answer = prediction["answers"][0]["answer"] if prediction["answers"] else "No answer found."

        # Display the response
        st.markdown(f"**Response:** {answer}")

if __name__ == "__main__":
    run_streamlit_app()
