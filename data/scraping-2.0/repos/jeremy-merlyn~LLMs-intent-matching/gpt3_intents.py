import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

with open('BE_UO_intents.txt', 'r') as file:
    PROMPT = file.read()

def get_intent():
    command = input("\nCommand: ")
    response = openai.Completion.create(
    model="text-curie-001",
    prompt= PROMPT + "\nCommand: " + command + "\nResponse:",
    temperature=0.5,
    max_tokens=128,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["Command:"]
    )
    print(response["choices"][0]["text"])

if __name__ == "__main__":
    for i in range(10):
        get_intent()