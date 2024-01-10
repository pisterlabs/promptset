from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  # model="gpt-3.5-turbo",
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
