"""
    Testing out Input Formats for OpenAI's GPT-3 API.

    Originally from:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
"""

import os
import openai

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Send API request to OpenAI
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Knock knock."},
#         {"role": "assistant", "content": "Who's there?"},
#         {"role": "user", "content": "Orange."},
#     ],
#     temperature=1,
#     max_tokens=2,
#     n=3,
# )
# print(response)

# API Request for Pirate
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant.",
#         },
#         {
#             "role": "user",
#             "content": "Write a welcome message for viewers of a YouTube video in the style of the pirate Blackbeard.",
#         },
#     ],
#     temperature=1,
#     n=2,
# )
# print(response)

# API Request for In-Context Learning
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a calculator.",
        },
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
        {
            "role": "assistant",
            "content": "The answer is 4.",
        },
        {
            "role": "user",
            "content": "What is 8 - 3?",
        },
        {
            "role": "assistant",
            "content": "The answer is 5.",
        },
        {
            "role": "user",
            "content": "What is 9 - 4?",
        },
    ],
    temperature=0,
)
print(response)