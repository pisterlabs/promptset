from openai import OpenAI
client = OpenAI(api_key="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")

response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(response.choices[0].message.content)