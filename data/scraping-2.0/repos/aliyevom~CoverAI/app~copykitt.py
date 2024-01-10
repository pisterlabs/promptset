import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

subject = "coffe"
prompt = f"Generate upbeat branding snippet for {subject}"

response = openai.Completion.create(engine="davinci-instruct-beta-v3", prompt=prompt, max_tokens=32)

print(response)
