# Import all required modules
import os
import time
from typing_extensions import Concatenate
from uuid import uuid4
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import openai
import pinecone

# Load environment variables
load_dotenv('.env')
# Use the environment variables for the API keys if available
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# SECTION 1: LOAD AND TOKENIZE THE DOCUMENT AND SPLIT IT INTO CHUNKS

# Instantiate ReadTheDocsLoader to load the documents from 'rtdocs'
loader = ReadTheDocsLoader('rtdocs')
docs = loader.load()

# Get tokenizer for the 'gpt-3.5-turbo' model from tiktoken
tokenizer_name = tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

# Define function to get the length of a tokenized text
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Instantiate RecursiveCharacterTextSplitter to split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# Split document content into chunks and store each chunk's metadata
chunks = []
for idx, page in enumerate(tqdm(docs)):
    content = page.page_content
    if len(content) > 100:
        url = page.metadata['source'].replace('rtdocs/', 'https://')
        texts = text_splitter.split_text(content)
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[i],
            'chunk': i,
            'url': url
        } for i in range(len(texts))])

# SECTION 2: INITIALIZE AN EMBEDDING MODEL FOR USE LATER

embed_model = "text-embedding-ada-002"
res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

# SECTION 3: CREATE AN INDEX 

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, enviroment="us-west1-gcp")

index_name = 'opentronsapi-docs'

# Create Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine'
    )
    # Wait for index to be initialized
    time.sleep(1)

# Connect to Pinecone index and view its stats
index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()

# SECTION 4: EMBED THE CHUNKS INTO THE INDEX

# Define batch size
batch_size = 100

# Create and insert embeddings in batches
for i in tqdm(range(0, len(chunks), batch_size)):
    # Define batch boundary
    i_end = min(len(chunks), i+batch_size)
    # Get metadata for the batch
    meta_batch = chunks[i:i_end]
    # Get IDs for the batch
    ids_batch = [x['id'] for x in meta_batch]
    # Get texts to encode
    texts = [x['text'] for x in meta_batch]
    # Create embeddings (handle RateLimitError if it occurs)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            time.sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    # Get embeddings
    embeds = [record['embedding'] for record in res['data']]
    # Clean up metadata
    meta_batch = [{'text': x['text'], 'chunk': x['chunk'], 'url': x['url']} for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # Upsert to Pinecone
    index.upsert(vectors=to_upsert)
