import openai
import os
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

# before we train
prompt = f"What is Walmart revenue in 2022?"
response = openai.Completion.create(
    model="text-babbage-001",
    prompt=prompt,
    max_tokens=300,
)
print("before...")
print(response['choices'][0]['text'])


prompt = f"What is Walmart revenue in 2022?"
response = openai.Completion.create(
    model="babbage:ft-altoformula-2023-03-14-23-43-46",
    prompt=prompt,
    max_tokens=300,
)
print("after...")
print(response['choices'][0]['text'])


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

training_data_set = [
    {
        "prompt": "What is Walmart revenue in 2022?",
        "completion": "$573 billion"
    },
    {
        "prompt": "What is Amazon revenue in 2022?",
        "completion": "$469 billion"
    },
]

total_token = 0
for data in training_data_set:
    for prompt, completion in data.items():
        total_token += num_tokens_from_string(string=prompt, encoding_name='gpt2') + \
            num_tokens_from_string(string=completion, encoding_name='gpt2')
print(total_token)
