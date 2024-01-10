import hashlib
import json
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import pinecone
import tiktoken
from tqdm.auto import tqdm


load_dotenv()

env_vars = [
    'OPENAI_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT'
]
os.environ.update({key: os.getenv(key) for key in env_vars})

openai.api_key = os.getenv('OPENAI_API_KEY')
embed_model = "text-embedding-ada-002"


# Initialize the loader and load documents
loader = UnstructuredFileLoader("/home/dan/langchain_projects/chunker/cal.html")
docs = loader.load()
print(len(docs))

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Initialize the MD5 hash object
m = hashlib.md5()

# Initialize the chunk list
documents = []

# Define the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=10,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)


# Process each document
for doc in tqdm(docs):
    url = doc.metadata['source'].replace('rtdocs/', 'https://')
    m.update(url.encode('utf-8'))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc.page_content)
    print(len(chunks))
    
    # Create an entry for each chunk
    for i, chunk in enumerate(chunks):
        documents.append({
            'id': f'{uid}-{i}',
            'url': url,
            'text': chunk,
            'chunk': i,
        })

# Print the total number of documents
print(len(documents))

# Save documents to a file
with open('train.jsonl', 'w') as f:
    for doc in documents:
        f.write(json.dumps(doc) + '\n')

# Read the documents from the file
with open('train.jsonl', 'r') as f:
    for line in f:
        documents.append(json.loads(line))


res = openai.Embedding.create(
    input=["/home/dan/langchain_projects/chunker/cal.html"
    ], engine=embed_model
)

index_name = 'cal'

# initialize connection to pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# check if index already exists
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='dotproduct'
    )
# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats


batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(documents), batch_size)):
    # find end of batch
    i_end = min(len(documents), i+batch_size)
    meta_batch = documents[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'source': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
