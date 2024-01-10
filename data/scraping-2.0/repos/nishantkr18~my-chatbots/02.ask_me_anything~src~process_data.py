"""
This module contains the code to:
1. Split the data into chunks (sentences).
2. Create vector embeddings of these sentences.
3. Store them in a vectorstore.
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def process_data(data: str):
    """
    The function that processes the data.
    """

    # Split into sentences
    text_splitter = CharacterTextSplitter(
        chunk_size=100, chunk_overlap=0, separator='.')
    texts = text_splitter.split_text(data)

    # Create vector embeddings and store in vectorstore.
    embedding = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_texts(texts, embedding)

    # vectorstore.as_retriever().get_relevant_documents(
    #     'Who is the most important character in the movie?')

    return vectorstore
