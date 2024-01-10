from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "user", "content": "そばの原材料を教えて"},
    ],
    max_tokens=100,
    temperature=1,
    n=2,
)

print(response.model_dump_json(indent=2))
