import openai
openai.api_key = "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"

query = input("Enter your Query: ")

prompt = {query}
response = openai.Completion.create(
    engine="text-curie-001",
    prompt=prompt,
    max_tokens=512,
    n=1,
    stop=None,
    temperature=1,
)

message = response.choices[0].text.strip()
print(message)