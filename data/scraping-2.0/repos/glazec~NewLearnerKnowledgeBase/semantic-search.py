import cohere
import pinecone
import numpy as np
import os

co = cohere.Client(os.environ["COHERE_KEY"])
query = "杭州美食"
pinecone.init(
    api_key=os.environ["PINECONE_KEY"],
    environment="gcp-starter",
)

index_name = "cohere-newlearner"
index = pinecone.Index(index_name)


# create the query embedding
xq = co.embed(
    texts=[query],
    model="embed-multilingual-v2.0",
    input_type="search_query",
    truncate="END",
).embeddings

print(np.array(xq).shape)

# query, returning the top 5 most similar results
res = index.query(xq, top_k=5, include_metadata=True)


for match in res["matches"]:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")

# cohere rerank
