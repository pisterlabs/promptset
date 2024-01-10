from langchain.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader, DirectoryLoader
from dotenv import load_dotenv 

load_dotenv()

##1. Loading Documents with LangChain

import requests
from bs4 import BeautifulSoup

# URL of the webpage to load
url = "https://analytickit.com/product/"

# Fetch the HTML content from the URL
response = requests.get(url)
html_content = response.text

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# PDF Loader Example
loader = PyPDFLoader("https://arxiv.org/pdf/2312.14804.pdf")
doc = loader.load()

# Text Loader Example
text_loader_kwargs = {'autodetect_encoding': True}
loader = DirectoryLoader('../aikit/content', show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Recursive Character Text Splitter Example
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, length_function=len, is_separator_regex=False)
split_text = text_splitter.split_text("Your text here")

##2. Splitting Documents
# Character Text Splitter for Independent Comments
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0, length_function=lambda x: 1, is_separator_regex=False)
split_docs = text_splitter.split_documents(docs)



## 3. Vector Stores and Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import numpy as np

# Initialize the OpenAI Embeddings model
embedding = OpenAIEmbeddings()


'''
A larger value (closer to 1) indicates that the embeddings are more similar, meaning the texts have more similar meanings or contexts.
A smaller value (closer to 0 or negative) suggests that the embeddings are less similar, meaning the texts are more different in meaning or context.
'''

# Two sample texts with similar meanings
text1_similar = 'The sun is shining brightly today.'
text2_similar = 'Today, the sunshine is very bright.'

# Convert texts to embeddings
emb1_similar = embedding.embed_query(text1_similar)
emb2_similar = embedding.embed_query(text2_similar)

# Calculate and print the distance (cosine similarity) between embeddings
distance_similar = np.dot(emb1_similar, emb2_similar)
print(f'Distance between similar embeddings: {distance_similar}')

# Two sample texts with different meanings
text1_different = 'The quick brown fox jumps over the lazy dog.'
text2_different = 'A well-baked apple pie can make any day better.'

# Convert texts to embeddings
emb1_different = embedding.embed_query(text1_different)
emb2_different = embedding.embed_query(text2_different)

# Calculate and print the distance (cosine similarity) between embeddings
distance_different = np.dot(emb1_different, emb2_different)
print(f'Distance between different embeddings: {distance_different}')


# Initialize a vector database to store embeddings
# Note: 'split_docs' should be a list of documents already processed.
persist_directory = 'vector_store'
vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory=persist_directory)



## 4. Retrieval Techniques
# Similarity Search
# Perform a similarity search in the vector database
query_docs = vectordb.similarity_search('query text', k=3)
print("Documents retrieved by similarity search:", query_docs)

# Perform an MMR search for more diverse results
query_docs_mmr = vectordb.max_marginal_relevance_search('query text', k=3, fetch_k=30)
print("Documents retrieved by MMR search:", query_docs_mmr)


## 5. LLM-Aided Retrieval
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Initialize the OpenAI language model
llm = OpenAI(temperature=0.1)

# Define metadata information for the retriever
metadata_field_info = [AttributeInfo(name="source", description="Description of the source field", type="string")]

# Description of the document content
document_content_description = "Description of document content"

# Initialize the Self Query Retriever
retriever = SelfQueryRetriever.from_llm(llm, vectordb, document_content_description, metadata_field_info, verbose=True)

# Retrieve documents using the LLM-aided retriever
docs = retriever.get_relevant_documents("Your query", k=5)
print("Documents retrieved using LLM-aided retrieval:", docs)

