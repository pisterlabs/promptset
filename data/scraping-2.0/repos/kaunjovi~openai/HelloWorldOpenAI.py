from openai import OpenAI

client = OpenAI(
  organization='org-VOnAf86STOfuNVLfUgAzuQYo',
)

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )
# print(completion.choices[0].message)

# client.embeddings.create(
#   model="text-embedding-ada-002",
#   input="The food was delicious and the waiter...",
#   encoding_format="float"
# )

# print(f"{client}")

# Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}
