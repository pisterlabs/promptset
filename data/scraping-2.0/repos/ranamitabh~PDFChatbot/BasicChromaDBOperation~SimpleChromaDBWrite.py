import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    documents = PyPDFLoader("/Users/amitabhranjan/IdeaProjects/Chat-with-PDF-Chatbot/docs/harry.pdf").load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
    texts = text_splitter.split_documents(documents)
    texts = [str(d) for d in texts]
    print(f"Split into {len(texts)} chunks")
    #create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    print(f"Creating embeddings. May take some minutes...")
    chroma_client=chromadb.HttpClient(host='localhost', port='8000')
    collection = chroma_client.create_collection(name="pdfChatbot1")
    collection = chroma_client.get_collection(name='pdfChatbot1')
    collection.add(
        ids=[f'doc_{i}' for i in range(len(texts))],
        documents=texts
    )

    print(f"Ingestion complete!")

if __name__ == "__main__":
    main()