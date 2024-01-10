"""
A script for embedding documents and performing similarity searches using the langchain library.

This script demonstrates the process of loading text documents, 
embedding them using OpenAI's language model embeddings, 
and then performing a similarity search to find relevant content. 
It uses Chroma from the langchain library for embedding 
and similarity search functionalities.

Features:
- Load text documents from a file.
- Split text into chunks based on character count.
- Embed documents using OpenAI's language model embeddings.
- Create a Chroma database for storing and searching embedded documents.
- Perform a similarity search in the Chroma database.

Usage:
Run the script to load documents from 'facts.txt', embed them, and perform a similarity search 
based on a query about the English language.
"""

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

load_dotenv()

# Initialize OpenAI embeddings.
embeddings = OpenAIEmbeddings()

# Configure a text splitter to divide text into chunks.
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

# Load and split text from 'facts.txt' file.
loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# Create a Chroma database from the embedded documents.
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")

# Perform a similarity search in the Chroma database.
results = db.similarity_search_with_score(
    "What is an interesting fact about the English language?", k=4
)

# Print the top 4 results from the similarity search.
for result in results:
    print(f"\n{result[1]}\n{result[0].page_content}")
