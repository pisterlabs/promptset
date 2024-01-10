# Importing necessary libraries
import pinecone
import openai
from loader_module import chunks

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key")
pinecone.deinit()

# Create Pinecone database
db_name = "pinecone_db"
pinecone.create_index(db_name, metric="cosine", shards=1)

# Initialize Pinecone Index
index = pinecone.Index(index_name=db_name)

def embed_chunks(chunks):
    """
    Function to embed chunks using OpenAI and store in Pinecone DB.
    """
    # Initialize OpenAI
    openai.api_key = "your-openai-api-key"

    # Loop through the chunks
    for chunk in chunks:
        # Create embeddings using OpenAI
        response = openai.Embed.create(model="text-davinci-002", texts=[chunk["text"]])
        # Store embeddings in Pinecone
        index.upsert(items={chunk["id"]: response["embeddings"][0]})
        
# Embed the chunks and store in Pinecone
embed_chunks(chunks)

# Export the Pinecone database
pinecone_db = index
