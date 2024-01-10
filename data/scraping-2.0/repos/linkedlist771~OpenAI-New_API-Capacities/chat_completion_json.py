from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={"type": "json_object"},
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Now I want you to score the sentence 'I like apples' on a scale of 1 to 10 with "
                                "output format as:"
                                "{'score': score}, for example, {'score': 0.5}"}
  ]
)
print(response.choices[0].message.content)