import openai
import sys

def read_api_key():
    with open('key.txt', 'r') as file:
        api_key = file.read().strip()
    return api_key

def complete_prompt(prompt):
    api_key = read_api_key()
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Retrieve the model's reply
    reply = response.choices[0].message.content
    return reply

# # Example usage
# prompt = "What is the capital of France?"
# completed_prompt = complete_prompt(prompt)

# print(completed_prompt)

for line in sys.stdin:

    completed_prompt = complete_prompt(line)

    print(completed_prompt)