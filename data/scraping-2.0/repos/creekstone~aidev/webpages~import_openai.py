import openai
import os
from dotenv import load_dotenv
load_dotenv()

# Construct the path to the Downloads folder
downloads_folder = os.path.expanduser('~/Downloads')

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAPI_KEY')

def save_response(response_text):
  # Determine the length of the response
  response_length = len(response_text)

  # If the response is less than 15 characters, use the entire response as the filename
  if response_length < 15:
    filename = response_text
  else:
    # Otherwise, use the first 15 characters of the response as the filename
    filename = response_text[:15]

  # Save the response to a file with the specified filename in the Downloads folder
  with open(os.path.join(downloads_folder, filename), 'a') as f:
    f.write(response_text)

text = """write two sentences that describe the beauty of a woman"""

# Use the completions API to generate text using the ChatGPT model
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=text,
    temperature=0.9,
    max_tokens=4040,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
)

# print out some stuff
print([0, 0, 2])

# Extract the generated text from the response
generated_text = response["choices"][0]["text"]

# Print the generated text
print(generated_text)

# Save the generated text to a file
save_response(generated_text)

def save_response(response):
  # Determine the length of the response
  response_length = len(response)

  # If the response is less than 15 characters, use the entire response as the filename
  if response_length < 15:
    filename = response
  else:
    # Otherwise, use the first 15 characters of the response as the filename
    filename = response[:15]

  # Save the response to a file with the specified filename in the Downloads folder
  with open(os.path.join(downloads_folder, filename), 'w') as f:
    f.write(response)


text = """write two sentences that describe the beauty of a woman"""

engine2 = "text-ada-002" # Supposed to be better, check email.

# Use the completions API to generate text using the ChatGPT model
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=text,
    temperature=0.9,
    max_tokens=4040,
    top_p=1.0,
    frequency_penalty=0,
    presence_penalty=0,
)

# Print the generated text
print(response["choices"][0]["text"])

# Create file with response
save_response(response)