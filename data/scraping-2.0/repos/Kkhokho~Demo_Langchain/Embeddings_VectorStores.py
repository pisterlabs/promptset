from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

os.environ["OPENAI_API_KEY"] = "API_KEY"


file_path = "C:\\Users\\DELL\\OneDrive - Hanoi University of Science and Technology\\Tài liệu\\Demo\\data.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
)

texts = text_splitter.create_documents([text_content])

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)

# Do a simple vector similarity search

query = "autoencoder?"
result = db.similarity_search(query)
print("Câu trả lời: \n");
print(result[0])
