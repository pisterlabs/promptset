from openai import OpenAI
client = OpenAI()

# send a request to OpenAI's API
# JSON output
response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  # specify a format of output to be a JSON object
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
print(response)
print(response.choices[0].message.content)


