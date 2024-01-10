import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

client = chromadb.PersistentClient(path="./db")

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

collection = client.get_collection(name="demo-dataset-600", embedding_function=embedding_function.embed_documents)

def query(prompt, n=10): 
    vectors = embedding_function.embed_documents([prompt])
    
    docs = collection.query(
        query_embeddings=vectors,
        n_results=n
    )
    files = [m["filename"] for m in docs["metadatas"][0]]
    return files

# if __name__ == "__main__": 
#     images = query("Show me the images of nature.", 10)
#     print(images)
    

