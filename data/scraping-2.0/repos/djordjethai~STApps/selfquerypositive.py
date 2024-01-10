import os
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from mojafunkcja import st_style, pinecone_stats
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader

st_style()


def main():
    # Ovaj kod ce biti koricen da se Pinecone index napuni tekstovima duzine do 1000 karaktera i metadata
    # ideja je da kasnije mozemo da pronadjemo tekstove odredjenog autora na odredjenu temu i
    # korisitmo za definisanje stila za pisanje tekstova
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    st.subheader('Upload Positive documents and metadata for Pinecone Index')
    # Initialize Pinecone
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
                  environment=os.environ["PINECONE_API_ENV"])

    # Use an existing Pinecone index
    index_name = "embedings1"  # replace with your index name
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Documents upload

    col1, col2 = st.columns(2)
    with col1:
        dokum = st.file_uploader(
            "Izaberite dokument/e", key="upload_file", type=['txt', 'pdf', 'docx'])

        # Get metadata from user
        with st.form(key='person_topic_form'):
            namespace = st.text_input('Enter namespace : ')
            source = st.text_input("Unesi izvor : ")
            person_name = st.text_input('Enter person name : ')
            topic = st.text_input('Enter topic  : ')
            submit_button = st.form_submit_button(label='Submit')

        docs = []
        pass_count = 0
        if person_name and topic and dokum and submit_button and namespace:

            if ".pdf" in dokum.name:
                loader = UnstructuredPDFLoader(
                    dokum.name, encoding="utf-8")
            else:
                # Creating a file loader object
                loader = UnstructuredFileLoader(
                    dokum.name, encoding="utf-8")

            data = loader.load()  # Loading text from the file

            # Split the document into smaller parts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(data)

            # Ask the user if they want to do OpenAI embeddings

            # Create the OpenAI embeddings
            st.write(f'Ucitano {len(texts)} tekstova')

            for pass_count, text in enumerate(texts):

                doc = Document(page_content=text.page_content, metadata={
                    "person_name": person_name, "topic": topic, "source": source, "chunk": pass_count, "url": source})
                # # Embed the document
                docs.append(doc)

            with st.spinner("Sacekajte trenutak..."):
                # Create Pinecone VectorStore from documents
                vectorstore = Pinecone.from_documents(
                    docs, embeddings, index_name=index_name, namespace=namespace
                )

            st.success('Data inserted successfully into the Pinecone index.')
    with col2:
        index = pinecone.Index('embedings1')
        pinecone_stats(index)


if __name__ == "__main__":
    main()


