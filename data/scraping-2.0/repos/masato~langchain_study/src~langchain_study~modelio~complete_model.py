from openai import OpenAI

client = OpenAI()

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="今日の天気がとても良くて、気分が",
    stop="。",
    max_tokens=100,
    n=2,
    temperature=0.5,
)

print(response.model_dump_json(indent=2))
