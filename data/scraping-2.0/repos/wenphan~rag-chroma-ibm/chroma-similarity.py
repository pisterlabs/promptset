# Ignore SSL just for dev
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Get data 
import os
import wget

filename = "state_of_the_union.txt"
url = "https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt"

if not os.path.isfile(filename):
	wget.download(url, out=filename)

# Load data as documents
from langchain.document_loaders import TextLoader
loader = TextLoader(filename)
documents = loader.load()

# Split documents
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Insert texts into vector store with embeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
chroma_vectorstore = Chroma.from_documents(texts, embeddings)

# Query
query = "What did the president say about Ketanji Brown Jackson"

# Similarity search
texts_sim = chroma_vectorstore.similarity_search(query, k=3)
print("Number of relevant texts: " + str(len(texts_sim)))

print("\n")
print("First 100 characters of relevant texts.")
for i in range(len(texts_sim)):
	print("Text " + str(i) + ": " + str(texts_sim[i].page_content[0:100]))
