import openai
import os
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv("src\GPT\.env")

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API key
openai.api_key = api_key

# Get the prompt from command line arguments
prompt = " ".join(sys.argv[1:])

# Generate text using the GPT-3 model
response = openai.Completion.create(
    model="text-davinci-003",  # Use the GPT-3 engine
    prompt=prompt,
    max_tokens=100#,
    #temperature=1,
    #top_p=1,
    #frequency_penalty=0,
    #presence_penalty=0
)

# Print the generated text
print(response.choices[0].text.strip())