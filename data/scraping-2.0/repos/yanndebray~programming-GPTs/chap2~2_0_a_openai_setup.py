from openai import OpenAI
client = OpenAI(api_key = 'sk-XXXX')
res = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Say this is a test",
)
print(res.choices[0].text)