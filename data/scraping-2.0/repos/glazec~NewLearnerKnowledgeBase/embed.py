import cohere
import pinecone
import numpy as np
import os

co = cohere.Client(os.environ["COHERE_KEY"])
# texts = [
#     "Hello from Cohere!",
#     "مرحبًا من كوهير!",
#     "Hallo von Cohere!",
#     "Bonjour de Cohere!",
#     "¡Hola desde Cohere!",
#     "Olá do Cohere!",
#     "Ciao da Cohere!",
#     "您好，来自 Cohere！",
#     "कोहेरे से नमस्ते!",
# ]
# read texts.txt and split by ===
with open("texts.txt", "r") as f:
    texts = f.read().split("\n===\n")


response = co.embed(
    texts=texts, model="embed-multilingual-v2.0", input_type="search_document"
)
embeds = response.embeddings  # All text embeddings
shape = np.array(embeds).shape
print(shape[0])

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="e252e7ac-1237-4f3b-97af-4a61ed91f847",
    environment="gcp-starter",
)

index_name = "cohere-newlearner"

# if the index does not exist, we create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=shape[1], metric="cosine")

# connect to index
index = pinecone.Index(index_name)

batch_size = 256

ids = [str(i) for i in range(shape[0])]
# create list of metadata dictionaries
meta = [{"text": text} for text in texts]

# create list of (id, vector, metadata) tuples to be upserted
to_upsert = list(zip(ids, embeds, meta))

for i in range(0, shape[0], batch_size):
    i_end = min(i + batch_size, shape[0])
    index.upsert(vectors=to_upsert[i:i_end])

# let's view the index statistics
print(index.describe_index_stats())
