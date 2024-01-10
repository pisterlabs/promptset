import os
from openai import OpenAI
client=OpenAI()

client.api_key = os.getenv("OPENAI_API_KEY")
response = client.models.list()
model_ids = [model.id for model in response.data]

result = "\n".join(model_ids)

print(result)