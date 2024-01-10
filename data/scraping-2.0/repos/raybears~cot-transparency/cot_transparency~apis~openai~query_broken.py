import openai
from dotenv import load_dotenv

load_dotenv()

with open("broken_prompt.txt") as fh:
    text = fh.read()


response = openai.Completion.create(
    model="text-davinci-002",
    prompt=text,
)

print(text)
print(response)
