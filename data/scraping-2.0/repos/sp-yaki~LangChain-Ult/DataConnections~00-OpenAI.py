import openai
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

# Set your OpenAI API key as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the prompt for the text generation
prompt = "Give me two reasons to learn OpenAI API with Python?"

# Define the parameters for the API request
model_engine = "text-davinci-003"
temperature = 0.7
max_tokens = 100

# Call the OpenAI API to generate text based on the prompt
response = openai.ChatCompletion.create(
  model=model_engine,
  prompt=prompt,
  temperature=temperature,
  max_tokens=max_tokens
)

# Print the generated text
print(response["choices"][0]["text"])
