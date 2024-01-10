import openai
import os
import json
import requests


def get_gpt_response(prompt):
    openai.api_key = "<<KEYHERE>>"
    model_engine = "text-davinci-002"
    prompt = f"{prompt.strip()}\nAI:"
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    message = completions.choices[0].text
    return message.strip()

def main():
    # get prompt from command line
    prompt= input("You: ")
    while prompt.lower() not in ["bye", "goodbye"]:
        response = get_gpt_response(prompt)
        print(f"AI: {response}")
        prompt= input("You: ")

if __name__ == "__main__":
    main()
