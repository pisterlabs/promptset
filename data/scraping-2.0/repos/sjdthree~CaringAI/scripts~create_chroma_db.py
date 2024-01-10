from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pickle
import os

# Initialize the text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
)
embeddings = OpenAIEmbeddings()

# Initialize an empty list to hold all texts
all_texts = []

# Loop over all PDF files in the directory
for filename in os.listdir("pdfs"):
    if filename.endswith(".pdf"):
        # Load the document
        loader = PyPDFLoader(f"pdfs/{filename}")
        docs = loader.load()

        # Split the document into texts
        texts = text_splitter.split_documents(docs)

        # Add the texts to the list
        all_texts.extend(texts)

with open('all_texts.pkl', 'wb') as f:
    pickle.dump(all_texts, f)

# Create Chroma object from all texts
db = Chroma.from_documents(all_texts, embeddings,  persist_directory="./chroma_db")
db.persist()