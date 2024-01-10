# Import the OpenAI module and os module for environment variable access.
from openai import OpenAI
import os

# Set the API key for the OpenAI API from an environment variable for security reasons.
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# Create an instance of the OpenAI class.
client = OpenAI()

# Make a call to the chat completion endpoint of the GPT model specifying:
# - The model version as "gpt-4-1106-preview".
response = client.chat.completions.create(
	model="gpt-4-1106-preview",
	messages=[
		{"role": "user", "content": "Who won the world series in 2020?"}
	]
)

# Print out the response from the model.
print(response.choices[0].message.content)