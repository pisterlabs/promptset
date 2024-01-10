# the code generates new images from text promts using the dalle-3 model.
# here we are using google colab and google drive to run and store the code.
!pip install langchain
!pip install openai
!pip install python-dotenv
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

from openai import OpenAI
import os

# Set your OpenAI API key
openai_api_key = 'type your opwnai api key'
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the OpenAI client
client = OpenAI()

# Get user prompt input
user_prompt = input("Enter your prompt: ")

# Make a request to generate an image from the prompt
response = client.images.generate(
    model="dall-e-3",
    prompt=user_prompt,
    size="1024x1024",  # Adjust size as needed
    quality="standard",
    n=1,
)

# Extract the generated image URL from the response
image_url = response.data[0].url

# Print the URL or use it as needed
print("Generated Image URL:", image_url)
