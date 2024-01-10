import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

index_name = "fleek-authority-index"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536
    )

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "Try wearing Allen Edmonds Men's Park Avenue Cap-Toe Oxfords. These black, classic leather shoes are handcrafted and made with high attention to detail. Their sleek, lace-up design adds a formal and quintessential look to any outfit."
docs = docsearch.similarity_search(query, 10)
print(len(docs))
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)