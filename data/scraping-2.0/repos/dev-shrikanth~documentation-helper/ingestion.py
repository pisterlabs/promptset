import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])
def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    print("Loading raw documents")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", ""])
    print("Splitting documents using splitter")
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} documents")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} into Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents, embedding=embeddings, index_name="langchain-doc-index")
    print("Vectors created in Pinecone")




if __name__ == "__main__":
    ingest_docs()
