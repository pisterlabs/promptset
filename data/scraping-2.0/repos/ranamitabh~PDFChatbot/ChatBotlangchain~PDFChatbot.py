"""
This script extracts text from a given URL, splits it into smaller chunks, and performs question answering using the FAISS library.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It is used in this script to create a vector store from the extracted text chunks. The vector store allows for fast similarity search, which is used to find relevant documents for a given query.

The script uses the langchain library, which provides various NLP functionalities such as document loading, text splitting, embeddings, and question answering. It also uses the Hugging Face library for pre-trained language models.

The main steps of the script are as follows:
1. Import necessary modules.
2. Define the URL to extract text from.
3. Define a function to extract text from the URL using BeautifulSoup.
4. Write the extracted text to a file.
5. Load the text from the file using the TextLoader from langchain.
6. Split the document into smaller chunks using RecursiveCharacterTextSplitter from langchain.
7. Create embeddings for the text chunks using HuggingFaceEmbeddings from langchain.
8. Create a vector store using FAISS from langchain.
9. Load a pre-trained language model from Hugging Face Hub using HuggingFaceHub from langchain.
10. Load a question answering chain using load_qa_chain from langchain.
11. Perform a similarity search using the query and the vector store.
12. Run the question answering chain on the retrieved documents and the query.
13. Print the result.
"""
#importing moudles
import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#url
url = "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"
def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup(['scripts']):
        script.extact()
    return soup.get_text().lower()

with open("ronaldo.txt", "w") as f:
    f.write(extract_text(url))

loader = TextLoader("ronaldo.txt")
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0, separators=[" ", ",", "\n"])
docs = text_splitter.split_documents(document)
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embedding)
# faiss.write_index(db.index, 'faiss_index.bin')
llm=HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature":0.2, "max_length":256}
)
chain = load_qa_chain(llm, chain_type="stuff")
# query = "when was cristiano ronaldo born ?"
query = "when did cristiano ronaldo played his first official match in a UEFA Champions League "
docs = db.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print(res)
# index = faiss.read_index('/Users/amitabhranjan/IdeaProjects/PDFChatbot/ChatBotlangchain/faiss_index.bin')
#
# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-small",
#     model_kwargs={"temperature": 0.2, "max_length": 256}
# )
# chain = load_qa_chain(llm, chain_type="stuff")
# # query = "when was cristiano ronaldo born ?"
# query = "when did cristiano ronaldo played his first official match in a UEFA Champions League "
# query_vector = np.array([query])  # Convert query to 2D numpy array
# query_vector = np.reshape(query_vector, (1, -1))  # Reshape query_vector to have 2 dimensions
# _, I = index.search(query_vector, k=5)  # Perform similarity search
# docs = [index.reconstruct(i) for i in I[0]]
# res = chain.run(input_documents=docs, question=query)
# print(res)