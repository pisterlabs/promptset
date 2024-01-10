import openai
import os
from dotenv import load_dotenv
load_dotenv()
# Construct the path to the Downloads folder
downloads_folder = os.path.expanduser('~/Downloads')

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAPI_KEY')

def get_unique_filename(filename):
  # Get the list of all files in the Downloads folder
  filenames = os.listdir(downloads_folder)
  
  # If the given filename is not in the list, return it as is
  if filename not in filenames:
    return filename

  # If the given filename is already in the list, create a new filename by appending
  # the next sequential integer to it
  i = 1
  while True:
    new_filename = f"{filename} ({i})"
    if new_filename not in filenames:
      return new_filename
    i += 1

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
  with open(os.path.join(downloads_folder, get_unique_filename(filename)), 'a') as f:
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

# Extract the generated text from the response
generated_text = response["choices"][0]["text"]

# Print the generated text
print(generated_text)

# Save the generated text to a file
save_response(generated_text)
