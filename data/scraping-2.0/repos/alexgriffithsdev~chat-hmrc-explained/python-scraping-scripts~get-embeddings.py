import os
import pinecone
import os
import openai
import nltk
import json
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")
nltk.download('punkt')


# Breaks large page of text into smaller chunks
def chunk_text(text):
    """Splits a text into smaller chunks (paragraphs) of approximately 2000 characters using NLTK's sentence tokenizer."""
    if len(text) <= 2000:
        return [text]

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Add the current sentence to the current chunk
        current_chunk += sentence + " "

        # If the current chunk is greater than 2000 characters, add it to the list of chunks and start a new chunk
        if len(current_chunk) > 2000:
            chunks.append(current_chunk)
            current_chunk = ""

    # Add the last chunk to the list of chunks
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Load scraped content and their links from the txt file
with open("content.txt", "r") as file:
    data = json.load(file)

# Connect to Pinecone vector database
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(os.getenv("PINECONE_INDEX"))

pineconeInserts = []
i = 0

# Go through each topic, get the content, embed it using OpenAI then add it to our Pinecone database
for topic in data:
    pages = topic["pages"]

    for key, value in pages.items():
        
        chunks = chunk_text(value)

        for chunk in chunks:
          embeddingRes = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
          embedding = embeddingRes['data'][0]['embedding']
          pineconeInserts.append((str(i), embedding['embedding'], { "topic": topic["topic"],"content_doc_id": str(embedding["linked_content_id"]) }))
          i += 1

index.upsert(vectors=pineconeInserts)
