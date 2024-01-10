from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# Initialize the NLP model and ChromaDB
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient()  # Assuming you want a persistent client
collection = chroma_client.get_collection("my_collection")  # Replace with your collection name

def vectorize_text(text):
    return model.encode(text)

def get_relevant_context(question, collection):
    question_vector = vectorize_text(question).tolist()  # Convert to list
    query_result = collection.query(
        query_embeddings=[question_vector],
        n_results=3,  # Adjust top_k as needed
        include=["documents"]  # Include the actual documents in results
    )

    # Check if 'results' key exists and if it has data
    if 'results' in query_result and query_result['results']:
        relevant_docs = [result['document'] for result in query_result['results']]
        return "\n".join(relevant_docs)
    else:
        return "No relevant documents found."

def get_openai_response(question, context, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    if not context.strip():
        return "No relevant information found to answer the question."
    prompt = context + "\nUser: " + question + "\nFogify Chatbot:"
    try:
        response = client.completions.create(model="text-davinci-003",
        prompt=prompt,
        max_tokens=150)
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

def main():
    openai_api_key = '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'  # Replace with your OpenAI API key
    
    while True:
        user_input = input("Ask the Fogify Chatbot: ")
        if user_input.lower() == 'exit':
            break
        context = get_relevant_context(user_input, collection)
        response = get_openai_response(user_input, context, openai_api_key)
        print("Fogify Chatbot:", response)

if __name__ == "__main__":
    main()
