import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the GPT-3 API key from environment variables
API_KEY = os.getenv("GPT_API_KEY")  

def gpt_completion(message: str):
    """
    Generate a completion using the GPT-4 model based on the given message.

    Args:
    - message (str): The user's input message.

    Returns:
    - str: The generated response from the GPT model.
    """
    
    # Set the API key for the openai library
    openai.api_key = API_KEY

    # Use the ChatCompletion method to generate a response using the GPT-4 model
    # The system message sets the context for the assistant and the user message is the prompt for the model
    return openai.ChatCompletion.create(
               model = "gpt-4-0613",
               temperature=0.5,
               max_tokens=280,
               messages = [
                   {"role": "system", "content": "Você é um assistente de clínica farmacêutica hospitalar."},
                   {"role": "user", "content": message},
               ]).choices[0].message.content
