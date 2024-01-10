from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def retrieve_most_relevant_chunk(query, sections):
    model = SentenceTransformer('paraphrase-distilroberta-base-v2')

    # Generate embeddings
    query_embeddings = model.encode(query)
    sections_embeddings = model.encode(sections)

    # Compute similarity
    similarities = cosine_similarity(query_embeddings, sections_embeddings)

    # Get the index of the most similar document
    retrieved_doc_id = np.argmax(similarities[0])

    return sections[retrieved_doc_id]

def engineering_prompt(query, content):
    # Split the content using the function
    sections = split_text_into_chunks(content, 1000, 200)
    
    relevant_section = retrieve_most_relevant_chunk(query, sections)
    print("Query:", query[0])
    print("\nMost relevant chunk to the query:\n")
    print(relevant_section)
    
    return relevant_section

# Load the text file
with open('data/hvac.txt', 'r', encoding='utf-8') as f:
    content = f.read()

while True:
    prompt = input("\nEnter a prompt (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break
    else:
        print("\nGenerating text...\n")
        relevant_section = engineering_prompt([prompt], content)
        print("\nGenerated text:\n")
        print(relevant_section)
