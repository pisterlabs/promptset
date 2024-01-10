import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client with your API key
client = openai.Client(api_key=openai_api_key)

# Create a completion request for a poem
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",  # Replace with your preferred engine
    prompt="tell me about react js with example code",  # Adjust the prompt as desired
    max_tokens=550  # Set the max tokens to generate the desired length of the poem
)

# Extract the generated poem text from the response
generated_poem = response.choices[0].text.strip()

# Replace "\n" with actual line breaks ("\n")
formatted_poem = generated_poem.replace("\\n", "\n")

# Print the formatted poem
print(formatted_poem)