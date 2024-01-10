import streamlit as st
import textwrap
import tempfile
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

def wrap_text_preserve_newlines(text, width=110):
    
    lines = text.split('\n')


    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

   
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def main():
    st.title("Document QA Chatbot")

    
    uploaded_file = st.file_uploader("Upload a text document", type=["txt"])

    if uploaded_file is not None:
        
        uploaded_text = uploaded_file.read().decode('utf-8')

       
        st.subheader("Uploaded Document:")
        st.text(wrap_text_preserve_newlines(uploaded_text))

        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
            temp_file.write(uploaded_text)
            temp_file_path = temp_file.name

        loader = TextLoader(file_path=temp_file_path)
        documents = loader.load()

        
        os.remove(temp_file_path)

       
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(docs, embeddings)

       
        user_question = st.text_input("Enter your question:")

        if st.button("Ask"):
            
            results = db.similarity_search(user_question)

            
            doc_text = " ".join(doc.text for doc in results)

            
            answer = qa_model(question=user_question, context=doc_text)

            
            st.subheader("Chatbot Answer:")
            st.write(f"Answer: {answer['answer']}")
            st.write(f"Score: {answer['score']:.4f}")

if __name__ == "__main__":
    main()



