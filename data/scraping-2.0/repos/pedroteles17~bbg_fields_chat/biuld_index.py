#%%
import pandas as pd
import dotenv
import time
from tqdm import tqdm

import chromadb
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI

dotenv.load_dotenv()

#%%
# Load Data
all_fields_data = pd.read_parquet('data/clean_fields_docs.parquet')

all_fields_data['documentation'] = all_fields_data['documentation']\
    .apply(lambda x: x.replace('\n', ' '))

documents = []
for i, row in all_fields_data.iterrows():
    document = Document(
        text=f"{row['description']}: {row['documentation']}",
        metadata={
            "field_mnemonic": row['mnemonic'],
            "field_description": row['description'],
            "field_category": row['category_name'],
            "source": row['source'],
        },
        excluded_embed_metadata_keys=["source"]
    )
    documents.append(document)

#%%
# Create storage_context (Persistent Storage, ChromaDB)
db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("bloomberg_fields")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create service_context (LLM, Embedding, Text Splitter)
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

llm = OpenAI(model="gpt-3.5-turbo-1106")

service_context = ServiceContext.from_defaults(
    llm=llm, text_splitter=text_splitter, 
)

#%%
# Create index
index = VectorStoreIndex(
    [], service_context=service_context, 
    storage_context=storage_context, use_async=False
)

for document in tqdm(documents, total=len(documents)):
    time.sleep(0.01)
    index.insert(document)
