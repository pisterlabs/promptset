from openai import OpenAI

client = OpenAI()

TEMPERATURE = 0.6

messages = [
  {"role": "system", "content": "Generate HTML, CSS and JavaScript for a website based on the user prompt."},
  {"role": "user", "content": "Create a commerce website."}
]

result = client.chat.completions.create(model="gpt-3.5-turbo-0613",
messages=messages,
temperature=TEMPERATURE)

print(result)