import os
import openai
from tokenizer import count_tokens

# Since is free, give error of too many requests of people

openai.api_key = os.getenv("OPENAI_API_KEY")

model = "code-davinci-002"
prompt = "Create a function that takes a string and returns a string with the first letter of each word capitalized."

count_tokens(model, prompt)

response = openai.Completion.create(
    model=model,
    prompt=prompt,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response.choices[0].text)