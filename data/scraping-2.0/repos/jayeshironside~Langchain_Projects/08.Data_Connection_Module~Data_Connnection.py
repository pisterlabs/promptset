# Its like a brigde between external data and model.
    # 1. Document Loaders
    # 2. Document Transformers
    # 3. Text Embedding Models
    # 4. Vector Stores
    # 5. Retrievers

# Data Connection:
# SOURCE > LOAD > TRANSFORM > EMBED > STORE > RETRIEVE

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Source - Where we have the external file. It can be of any format.
# - In our case we have the Sample.txt in the same folder


# Document Loaders - Load document from different sources
loader = TextLoader('Sample.txt')
documents = loader.load()

print(len(documents))

# Document Transformers - Split document and drop redundant document
text_splitter = CharacterTextSplitter(chunk_size = 200, chunk_overlap = 0)
texts = text_splitter.split_documents(documents)

print(len(texts))


# Text Embedding models - Take unstructured text and turn it into a list of floating numbers.
embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')

# Vector Stores - Store and Search over embeded data
    # Load embeddings of text into chroma/FAISS
db = FAISS.from_documents(texts, embeddings)
    # Lets have a look at embeddings - Numeric representation.
    # db._collection.get(include=['embeddings']) - use this if you use chromadb


# Retrievers - Query your data
retriever = db.as_retriever(search_kwags={"k":1})
print(retriever)

# Question - 1
docs = retriever.get_relevant_documents("What is the capital of india ?")
print(docs)