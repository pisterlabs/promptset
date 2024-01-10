from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    seed=123,
    max_tokens=200,
    temperature=0.7,
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "怖い夢を見ないようにする方法は?"}
    ]
)
print(response.choices[0].message.content)

print(response.system_fingerprint)
