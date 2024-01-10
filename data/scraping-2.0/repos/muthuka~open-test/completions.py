from openai import OpenAI
client = OpenAI()

messages1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant",
     "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages1,
)

print(response)

messages2 = [
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020 and where?"},
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=messages2,
)

print(response.choices[0].message.content)
