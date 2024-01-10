import os
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings

# Ovaj kod ce biti koricen da se Pinecone index napuni tekstovima duzine do 1000 karaktera i metadata
# ideja je da kasnije mozemo da pronadjemo tekstove odredjenog autora na odredjenu temu i
# korisitmo za definisanje stila za pisanje tekstova

# Retrieving API keys from env
open_api_key = os.environ.get('OPENAI_API_KEY')
# Initialize Pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_API_ENV"])
# Create a new namespace
namespace = "stilovi"  # replace with your namespace
# pinecone.create_namespace(namespace)
# Use an existing Pinecone index
index_name = "embedings1"  # replace with your index name
# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Streamlit app
st.title('Upload txt files and metadata')
# Documents upload
uploaded_files = st.file_uploader(
    "Choose a .txt file", type="txt", accept_multiple_files=True)

# Get metadata from user
with st.form(key='person_topic_form'):
    person_name = st.text_input('Enter person name : ')
    topic = st.text_input('Enter topic  : ')
    submit_button = st.form_submit_button(label='Submit')

docs = []
if person_name and topic:
    for uploaded_file in uploaded_files:
        # Read the file
        file_content = uploaded_file.getvalue().decode()
        doc = Document(page_content=file_content, metadata={
            "person_name": person_name, "topic": topic})
        # # Embed the document
        docs.append(doc)
    # st.write(docs)
    with st.spinner("Sacekajte trenutak..."):
        # Create Pinecone VectorStore from documents
        vectorstore = Pinecone.from_documents(
            docs, embeddings, index_name=index_name, namespace=namespace
        )

    st.success('Data inserted successfully into the Pinecone index.')


