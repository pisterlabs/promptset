import openai
openai.api_key = "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"
prompt = "Hello, World!"
response = openai.Completion.create(
  #davinci-codex
  #text-davinci-002
  engine="gpt-3.5-turbo",
  prompt=prompt,
  temperature=0.5,
  max_tokens=60,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response)
