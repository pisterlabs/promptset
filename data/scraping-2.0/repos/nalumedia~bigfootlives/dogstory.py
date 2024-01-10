import openai

openai.api_key = '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'

response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="This is a story about a dog named Avery. Avery is a {breed} with a {temperament} temperament.",
  max_tokens=150
)

print(response.choices[0].text.strip())
