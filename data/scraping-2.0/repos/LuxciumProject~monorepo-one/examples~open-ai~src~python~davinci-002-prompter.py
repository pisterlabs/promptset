import os

import openai

try:
    api_key = os.environ['KILO_TOKENS']
except KeyError:
    print("Error: The 'KILO_TOKENS' environment variable is not set.")
    exit(1)

openai.api_key = api_key

PROMPT = "Please generate a Python code snippet that converts a list of Fahrenheit temperatures to Celsius temperatures, with a temperature format of '{temperature}°F', and a maximum length of 1000 tokens."
MODEL_ENGINE = "text-davinci-002"

try:
    completions = openai.Completion.create(
        engine=MODEL_ENGINE, prompt=PROMPT, max_tokens=1000, n=1)
    output = completions.choices[0].text.strip()
    print(output)
except Exception as e:
    print(f"Error: {e}")
