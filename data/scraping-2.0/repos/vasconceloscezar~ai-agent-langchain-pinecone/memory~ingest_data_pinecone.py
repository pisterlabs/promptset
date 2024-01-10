from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from testing_pinecone import init_pinecone

init_pinecone()

# Step 3: Create Pinecone index
index = pinecone.Index("gptest")

# Step 4: Initialize embeddings
embeddings = OpenAIEmbeddings()

# Step 5: Create Pinecone instance
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Step 6: Read local data from files
texts = []
with open("output.txt", "r") as f:
    texts.append(f.read())
# with open("file2.txt", "r") as f:
#     texts.append(f.read())
# ... add more files as needed

# Step 7: Embed local data and add it to the Pinecone index
vectorstore.add_texts(texts)
