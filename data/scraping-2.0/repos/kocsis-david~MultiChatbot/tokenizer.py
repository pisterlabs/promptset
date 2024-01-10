# This snippet shows and example how to use the Cohere Embed V3 models for semantic search.
# Make sure to have the Cohere SDK in at least v4.30 install: pip install -U cohere
# Get your API key from: www.cohere.com
import cohere
import numpy as np

cohere_key = "{YOUR_COHERE_API_KEY}"   #Get your API key from www.cohere.com
co = cohere.Client(cohere_key)

docs = ["The capital of France is Paris",
        "PyTorch is a machine learning framework based on the Torch library.",
        "The average cat lifespan is between 13-17 years"]


#Encode your documents with input type 'search_document'
doc_emb = co.embed(docs, input_type="search_document", model="embed-english-v3.0").embeddings
doc_emb = np.asarray(doc_emb)


#Encode your query with input type 'search_query'
query = "What is Pytorch"
query_emb = co.embed([query], input_type="search_query", model="embed-english-v3.0").embeddings
query_emb = np.asarray(query_emb)
query_emb.shape

#Compute the dot product between query embedding and document embedding
scores = np.dot(query_emb, doc_emb.T)[0]

#Find the highest scores
max_idx = np.argsort(-scores)

print(f"Query: {query}")
for idx in max_idx:
  print(f"Score: {scores[idx]:.2f}")
  print(docs[idx])
  print("--------")
