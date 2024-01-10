import os
import openai
from reader import read as read
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("GPT_KEY")

with open ("prompt.txt", "r") as myfile:
    prompt=myfile.read().replace('\n', '')

chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                               messages=[{"role": "user", "content": prompt}])
with open("output.txt", "w+") as text_file:
    print(chat_completion.choices[0].message, file=text_file)

read()
