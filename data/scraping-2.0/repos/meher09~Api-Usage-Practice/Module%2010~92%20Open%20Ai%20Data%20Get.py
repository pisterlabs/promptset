import openai

openai.api_key = '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'
prompt = input("Enter your Command ")
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

text = response.get('choices')[0].get('text')
print(text)