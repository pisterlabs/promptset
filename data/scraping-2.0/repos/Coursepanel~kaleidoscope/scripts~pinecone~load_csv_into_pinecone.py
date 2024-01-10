# Reference - https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pinecone.html
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import CSVLoader
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

loader= CSVLoader(file_path="/Users/ankitsanghvi/Desktop/kaleido_gpt/data/nptel-complete.csv")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
docs = text_splitter.split_documents(tqdm(documents, desc="Splitting documents", unit="doc"))

# Model used for embeddings is OpenAI's default text-embedding-ada-002 model with 1536 dimensions
# Reference link - https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
embeddings = OpenAIEmbeddings()
# initialize pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment="northamerica-northeast1-gcp" # next to api key in console
)
index_name = "nptel"
pinecone_index = pinecone.Index(index_name=index_name)
# Fetch existing document ids
# existing_doc_ids = list(pinecone_index.fetch_document_ids())
# doc_ids_set = set(existing_doc_ids)

print("Creating 🍍 index...")
docsearch = Pinecone.from_documents(tqdm(docs, desc="Indexing documents", unit="doc"), embeddings, index_name=index_name)
print("🍍 index created ! 🎉...")

# if you already have an index, you can load it like this
# print(f"Putting 🍍 data into index {index_name}...")

# #? Old code
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

#? New code
# docsearch = Pinecone(embeddings=embeddings, index=pinecone_index)
# for doc in tqdm(docs, desc="Adding documents to existing index", unit="doc"):
#     doc_id = doc.identifier
#     if doc_id not in doc_ids_set:
#         docsearch.add_document(doc)

query = "Which is the best course for machine learning?"
print("Searching for similar documents to: ", query)
docs = docsearch.similarity_search(query)
print(docs)