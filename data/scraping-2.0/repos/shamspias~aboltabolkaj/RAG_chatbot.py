import openai
import pinecone
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class RAGChatbot:
    def __init__(self, openai_key, pinecone_key, pinecone_environment, pinecone_index):
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_key, environment=pinecone_environment)
        self.index = pinecone.Index(pinecone_index)

        # Initialize OpenAI
        openai.api_key = openai_key

        # Initialize chat history
        self.chat_history = []

    def embed_text(self, text):
        # Embed the text using OpenAI's text-embedding model
        response = openai.Embedding.create(
            input=[text],
            model="text-similarity-babbage-001"  # Choose the appropriate embedding model
        )
        # Extract the embedding vector
        return response['data'][0]['embedding']

    def get_similar_data(self, question_embedding, top_k=1):
        # Retrieve similar data from Pinecone
        try:
            results = self.index.query(queries=[question_embedding], top_k=top_k, include_metadata=True)
            return results
        except Exception as e:
            print(f"An error occurred while querying Pinecone: {e}")
            return None

    def generate_response(self, question, similar_data):
        # Generate a response using OpenAI's GPT-3.5
        context = "\n".join([item['metadata']['text'] for item in similar_data['matches']])
        prompt = f"The following are some relevant pieces of information:\n{context}\n\nBased on the above, answer the question: {question}"
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"An error occurred while generating a response from OpenAI: {e}")
            return None

    def chat(self, question):
        # Embed the question text
        question_embedding = self.embed_text(question)
        
        # Get similar data from Pinecone
        similar_data = self.get_similar_data(question_embedding)
        if similar_data is None:
            return "I'm sorry, I couldn't retrieve similar data at the moment."

        # Generate a response from GPT-3.5 using the similar data
        response = self.generate_response(question, similar_data)
        if response is None:
            return "I'm sorry, I couldn't generate a response at the moment."

        # Save to chat history
        self.chat_history.append({"question": question, "response": response})
        
        return response

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT_NAME = os.getenv('PINECONE_ENVIRONMENT_NAME')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create an instance of the chatbot
rag_chatbot = RAGChatbot(
    openai_key=OPENAI_API_KEY,
    pinecone_key=PINECONE_API_KEY,
    pinecone_environment=PINECONE_ENVIRONMENT_NAME,
    pinecone_index=PINECONE_INDEX_NAME
)

# Example interaction
if __name__ == "__main__":
    while True:
        user_question = input("You: ")
        if user_question.lower() in ['exit', 'quit']:
            break
        bot_response = rag_chatbot.chat(user_question)
        print(f"Bot: {bot_response}")
