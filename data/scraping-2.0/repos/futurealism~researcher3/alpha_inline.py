import openai
import pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI and Pinecone
openai.api_key = openai_api_key
pinecone.init(api_key=pinecone_api_key, environment='us-west1-gcp')

# Setup Pinecone Index
index_name = "ideas"
pinecone.create_index(index_name, dimension=2048, metric='cosine')
index = pinecone.Index(index_name)

def encode_idea(idea_text):
    """
    Encodes the given idea text using Ada-2 and returns the vector.
    """
    response = openai.Embedding.create(
        input=idea_text,
        engine="ada-2-embedding",
        n=1
    )
    vector = response['data'][0]['embedding']
    return vector

def process_idea(idea_text):
    """
    Processes the idea using GPT-4 for expansion and refinement.
    """
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=idea_text,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def store_idea(idea_id, idea_text, vector):
    """
    Stores the idea and its vector in Pinecone.
    """
    index.upsert(vectors={idea_id: vector}, metadata={"text": idea_text})

def retrieve_similar_ideas(vector):
    """
    Retrieves similar ideas based on the vector from Pinecone.
    """
    results = index.query(vector, top_k=5)
    return results['matches']

def main():
    user_idea = input("Enter your idea: ")
    processed_idea = process_idea(user_idea)
    idea_vector = encode_idea(processed_idea)
    idea_id = "idea_" + str(hash(processed_idea))  # Generating a unique ID for the idea

    store_idea(idea_id, processed_idea, idea_vector)
    similar_ideas = retrieve_similar_ideas(idea_vector)

    print("Your processed idea:", processed_idea)
    print("Similar ideas:", similar_ideas)

if __name__ == "__main__":
    main()
