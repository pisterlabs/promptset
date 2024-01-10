import os
import openai

# api_type, api_base and api_version required when using Azure OpenAI service
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"

# api_key always required for Azure OpenAI service and OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# engine parameter required when using Azure OpenAI service
response = openai.ChatCompletion.create(
    engine="chat",
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you say something to inspire the audience of Microsoft BUILD 2023?"}
    ]
)

print(response["choices"][0]["message"]["content"])
