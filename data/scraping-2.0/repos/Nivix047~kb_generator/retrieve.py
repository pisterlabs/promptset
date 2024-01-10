import os
import dotenv
import openai
import pinecone

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path=".env")

# Function to retrieve text based on a query using Pinecone
def retrieve(query, index):
    try:
        # Create text embeddings for the query
        res = openai.Embedding.create(input=[query], engine="text-embedding-ada-002")
        xq = res['data'][0]['embedding']
        # Query the Pinecone index with the embeddings
        res = index.query(xq, top_k=1, include_metadata=True)
        if res['matches']:
            # If matches found, prepare a prompt with context
            context = res['matches'][0]['metadata']['text']
            prompt = f"Answer the question based on the context below.\n\ncontext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            return prompt
        else:
            # Return a message if no relevant information is found
            return "No relevant information found in the index for the given query."
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return None

# Function to complete a given prompt using GPT-3.5-Turbo-Instruct
def complete(prompt):
    try:
        if prompt:
            # Generate a completion based on the prompt
            response = openai.Completion.create(engine='gpt-3.5-turbo-instruct', prompt=prompt, max_tokens=400)
            return response['choices'][0]['text'].strip()
        else:
            # Return a message if the prompt is missing
            return "No completion available due to missing prompt."
    except Exception as e:
        print(f"An error occurred during completion generation: {e}")
        return None

# Main function to execute the script
def main():
    try:
        # Initialize OpenAI and Pinecone with API keys
        openai.api_key = os.getenv("OPENAI_API_SECRET")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please check your .env file.")

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not found. Please check your .env file.")
        pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")

        # Retrieve Pinecone index name from environment variables
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("Pinecone index name not found in environment variables.")

        # Load the Pinecone index
        index = pinecone.Index(index_name=index_name)

        # Example usage of the retrieve and complete functions
        query = "What do $gt and $gte do in MongoDB?"
        query_with_context = retrieve(query, index)
        completion = complete(query_with_context)
        if completion:
            print(completion)
        else:
            print("No completion was generated.")

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Execute the main function
if __name__ == "__main__":
    main()
