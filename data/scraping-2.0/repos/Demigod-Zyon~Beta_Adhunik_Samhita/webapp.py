import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os

st.title("Document Search Web App")

# Load PDFs and create the index
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
index = VectorstoreIndexCreator().from_loaders(loaders)

# User input
query = st.text_input("Enter your query here:")

if st.button("Search"):
    if query:
        results = index.query(query)
        st.write("Search Results:")
        for result in results:
            st.write(result)
    else:
        st.warning("Please enter a query.")

if __name__ == "__main__":
    st.run()
