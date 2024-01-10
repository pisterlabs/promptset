import os

import openai

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "engage in a helpful, detailed, polite conversation with a user."
        ),
    },
    {
        "role": "user",
        "content": ("Count to 100, with a comma between each number and no newlines. " "E.g., 1, 2, 3, ..."),
    },
]

# demo chat completion without streaming
# models https://docs.perplexity.ai/docs/model-cards
response = openai.ChatCompletion.create(
    model="mistral-7b-instruct",
    messages=messages,
    api_base="https://api.perplexity.ai",
    api_key=PERPLEXITY_API_KEY,
)
print(response)

# demo chat completion with streaming
# response_stream = openai.ChatCompletion.create(
#     model="mistral-7b-instruct",
#     messages=messages,
#     api_base="https://api.perplexity.ai",
#     api_key=PERPLEXITY_API_KEY,
#     stream=True,
# )
# for response in response_stream:
#     print(response)
