import openai
from openai import OpenAI
import pinecone
import numpy as np

class TextEmbeddingQuery:
    '''
    This class uses OpenAI to convert user input text into embeeding,
    query the Pinecone Index to get the matching result. The embeedings are
    uploaded to PineCone using class EmbeddingUploader
    '''
    def __init__(self, pinecone_api_key, pinecone_env, index_name, openai_api_key, model_name="text-embedding-ada-002"):
        # Initialize Pinecone with the provided API key and environment, and create an index object.
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        self.index = pinecone.Index(index_name)

        # Initialize the OpenAI client with the provided API key and set the model name for embeddings.
        openai.api_key = openai_api_key
        self.client = OpenAI()
        self.model_name = model_name

    def generate_embedding(self, text):
        # Generate an embedding for the given text using the specified OpenAI model.
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    def query(self, user_input):
        # Generate an embedding for the user input and query the Pinecone index for similar items.
        user_embedding = self.generate_embedding(user_input)

        # Convert the embedding to a list if it's a numpy array.
        if isinstance(user_embedding, np.ndarray):
            user_embedding = user_embedding.tolist()

        # Perform the query on the Pinecone index.
        query_result = self.index.query(
            vector=user_embedding,
            top_k=3,
            include_metadata=True
        )

        # Format and return the query results.
        result_text = "Query Results:\n"
        for i, match in enumerate(query_result['matches']):
            snippet = match['metadata'].get('text', '')
            result_text += f"Match {i+1}: ID = {match['id']}, Score = {match['score']:.6f}, Snippet: {snippet}\n"

        return result_text

# For local testing
if __name__ == "__main__":
    # Initialize the TextEmbeddingQuery class with API keys and index name.
    pinecone_api_key = "your-pinecone-api-key"
    pinecone_env = "gcp-starter"
    pinecone_index = "kit"
    openai_api_key = 'your-openai-api-key'

    teq = TextEmbeddingQuery(pinecone_api_key, pinecone_env, pinecone_index, openai_api_key)

    # Test the query method with a sample input.
    sample_input = "How Secure Are Our Data?"
    result = teq.query(sample_input)
    print(result)
