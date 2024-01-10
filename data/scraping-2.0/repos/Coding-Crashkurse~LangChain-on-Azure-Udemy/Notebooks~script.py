import requests
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

loader = DirectoryLoader("./restaurant", glob="**/*.txt", loader_cls=TextLoader)
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
docs = text_splitter.split_documents(data)

documents_to_send = [
    {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
]
print(documents_to_send)

url = "http://localhost:5000/index_documents"

try:
    response = requests.post(url, json=documents_to_send)
    response.raise_for_status()
    print("Response from server:", response.json())
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
