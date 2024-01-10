from langchain.document_loaders.json_loader import JSONLoader
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
load_dotenv()
# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    metadata["tags"] = record.get("tags")
    metadata["images_list"] = record.get("images_list")
    metadata["handle"] = record.get("handle")
    return metadata

def create_vectorstore(documents, embeddings):
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore

def save_vectorstore(vectorstore, save_path, index_name):
    vectorstore.save_local(save_path, index_name)
    print("Vectorstore saved to: ", save_path)

loader = JSONLoader(
    file_path='./products.json',
    jq_schema='.[]',
    content_key="expanded_description",
    metadata_func=metadata_func
)

if __name__ == "__main__":
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = create_vectorstore(documents, embeddings)
    save_vectorstore(vectorstore, save_path="./shopify_langchaintesting_vectorstore", index_name="products")

