import os
import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import sys
from tenacity import retry, stop_after_attempt, wait_exponential

# The following line increases the maximum recursion limit in Python.
# This is necessary because the RecursiveCharacterTextSplitter later in the code may
# cause a RecursionError if the recursion limit is not increased.
sys.setrecursionlimit(3000)

# The program operates relative to the location of this script file.
# So we start by getting the directory that this script file is in.
base_dir = os.path.dirname(os.path.realpath(__file__))

# We then move two levels up in the directory structure. This is the project's root directory.
grandparent_dir = os.path.dirname(base_dir)

# OpenAI needs an API key to work, and we load that from a secret st file
openai.api_key = st.secrets["OPENAI_API_KEY"]

# We will be storing some data in a chroma (Facebook AI Similarity Search) index.
# If a previous run of this script has already created that index, we want to delete it before creating a new one.
chroma_index_path = os.path.join(grandparent_dir, 'chroma_index')

try:
    if os.path.exists(chroma_index_path):
        for file_name in os.listdir(chroma_index_path):
            file_path = os.path.join(chroma_index_path, file_name)
            os.remove(file_path)
except PermissionError:
    print(f"No permission to delete files in {chroma_index_path}. Please check the file permissions.")

# Next, we load up some documents. There are two types of documents we're interested in.
# content from a Notion document and Markdown content that's stored in separate directories.

# Load the Notion document content from 'content/notion' directory
notion_loader = NotionDirectoryLoader(os.path.join(grandparent_dir, 'content', 'notion'))
notion_documents = notion_loader.load()

# Load the markdown content from 'content/blogs' directory
markdown_loader = DirectoryLoader(os.path.join(grandparent_dir, 'content', 'blogs'))
blog_documents = markdown_loader.load()

# Load another markdown content from 'content/docs' directory
markdown_loader = DirectoryLoader(os.path.join(grandparent_dir, 'content', 'docs'))
guide_documents = markdown_loader.load()

# Combine all loaded documents together
documents = notion_documents + blog_documents + guide_documents

# Next, these documents are split into smaller chunks for processing.
# The split points are defined by a set of separators (any level of a markdown header, newlines and periods),
# and each chunk is limited by a size (1500 characters) with some overlap (100 characters).
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###", "\\n\\n", "\\n", "."],
    chunk_size=1500,
    chunk_overlap=100
)

# Splits all the loaded documents
docs = markdown_splitter.split_documents(documents)

# The OpenAI Embeddings model is used to convert these chunks into embeddings (vector representations of the text).
embeddings = OpenAIEmbeddings()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=5, max=20))
def create_chroma_from_documents(doc, embeddings, chroma_index_path):
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=chroma_index_path)
    return vectordb


# Convert documents into vector embeddings and store these vectors into a chroma index.
vectordb = create_chroma_from_documents(docs, embeddings, chroma_index_path)

print('Local Chroma index has been successfully saved.')
