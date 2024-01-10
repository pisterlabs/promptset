import openai
from apikey import gpt_apikey

openai.api_key = gpt_apikey()

prompt = "Generate a spam email:"
model = "text-davinci-002"
temperature = 0.5
max_tokens = 30

response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
)

spam = response.choices[0].text.strip()
print(spam)