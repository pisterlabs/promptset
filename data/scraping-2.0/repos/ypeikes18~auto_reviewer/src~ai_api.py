import openai
import os

openai.api_key=os.getenv("OPEN_AI_API_KEY")

# TODO move to config file
LETTERS_PER_TOKEN = 4

def get_response_to_prompt(prompt):
    completion = openai.ChatCompletion.create(model="gpt-4", messages = [{"role": "assistant", "content": prompt}])
    return completion.choices[0].message.content


def approximate_tokens(prompt: str):
    return len(prompt)/LETTERS_PER_TOKEN
