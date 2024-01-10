from dotenv import load_dotenv
import os
import openai

# Load the environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the prompt from the text file
with open('data/prompt-email.txt', 'r', encoding='utf-8') as file:
    prompt_prep = file.read().strip()

# Load the prompt from the text file
with open('data/prompt-test.txt', 'r', encoding='utf-8') as file:
    prompt_content = file.read().strip()

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {
      "role": "user",
      "content": prompt_prep + prompt_content
    }
  ],
  temperature=0,
  max_tokens=4096,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

# Print the response from OpenAI
print(response.choices[0].message['content'])
