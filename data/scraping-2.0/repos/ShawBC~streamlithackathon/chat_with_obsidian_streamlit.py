import streamlit as st
import zipfile
import os

# 1. Import necessary libraries
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import ObsidianLoader
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub

# 2. Set up Streamlit UI elements
st.title("Obsidian Query Interface")

# Ask user for huggingface api key
HUGGINGFACE_API_KEY = st.sidebar.text_input("HUGGINGFACE API Key", type="password")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACE_API_KEY

# get repo id from Hugging Face
repo_id = "google/flan-ul2"

# Upload zipped Obsidian vault
uploaded_vault = st.file_uploader("Upload a zipped Obsidian vault", type="zip")

# Extract the vault if uploaded
if uploaded_vault:
    vault_dir = "/tmp/obsidian_vault"
    with zipfile.ZipFile(uploaded_vault, 'r') as zip_ref:
        zip_ref.extractall(vault_dir)
    st.success("Vault uploaded and extracted successfully!")


# Enter a query
query = st.text_input("Enter your query:")


temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
max_length = st.sidebar.slider("Max Length", min_value=50, max_value=5000, value=2048, step=12)

# Button to execute query
if st.button("Execute Query"):
    if query and os.path.exists(vault_dir):
        # 3. Load the model and data using the provided code logic
        loader = ObsidianLoader(vault_dir)
        embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
        
        callbacks = [StreamingStdOutCallbackHandler()]


        llm = HuggingFaceHub(repo_id=repo_id, 
                             callbacks=callbacks, 
                             verbose=True, 
                             model_kwargs={"temperature": temperature, "max_length": max_length })


        index = VectorstoreIndexCreator(embedding=embeddings,
                                vectorstore_cls=Chroma, 
                                vectorstore_kwargs={"persist_directory": "db"}, 
                                text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders([loader])

        # 4. Process the query
        chain = RetrievalQA.from_chain_type(llm=llm, 
                    chain_type="stuff", 
                    retriever=index.vectorstore.as_retriever(),
                    input_key = "question")

        # 5. Display the results
        st.write("Results:", chain.run(query))
    else:
        st.warning("Please upload your obsidian zip file, and enter a query.")

