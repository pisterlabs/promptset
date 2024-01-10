from openai import OpenAI
import os

import random
from tenacity import retry, stop_after_attempt, wait_exponential
OPENAI_API_KEY=os.getenv("key")

client = OpenAI(api_key=OPENAI_API_KEY
                )
prompt = "you are an ai model"
temperature = .4
number_of_examples = 1
N_RETRIES = 3

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": "You are helping me create a dataset for indian receipes"
        },
        {
            "role": "user",
            "content": "give me 3 different Indian receipe"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )

    return response.choices[0].message.content

# Generate examples
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    print(example)
    prev_examples.append(example)