# import modules
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    # Load PDF data using PyPDFLoader
    data = PyPDFLoader('SAConstitution-web-eng.pdf').load()

    # Split the text into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(data)

    # Initialize the embeddings model for document matching
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

    # Generate a knowledge base from the split texts using the embeddings model
    knowledge_base = FAISS.from_documents(texts, embeddings)

    # Save the knowledge base locally to use later for document matching
    knowledge_base.save_local('knowledge_base')

if __name__ == '__main__':
    main()