from dotenv import load_dotenv
import os
import openai


prompt = "tell me a slogan for a home secutity company"


# Load environment variables from the .env file
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

output = openai.Completion.create(
    model='text-davinci-003',
    prompt=prompt,
    max_tokens=200,
    temperature=0
)

output_text = output['choices'][0]['text']

print(output_text)
