#!/usr/bin/env python

import openai
import os
import json

# Check that the API key was provided
if "OPENAI_API_KEY" not in os.environ:
    print("Error: Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# Set up OpenAI API client
openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "text-davinci-003"

def send_query(input, context):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=f"{context}{input}",
        max_tokens=4000,
        temperature=0,
    )

    if response["object"] != "text_completion":
        print("Error: Request failed")
        exit(1)

    response_text = response["choices"][0]["text"].strip()

    if not response_text:
        print("Error: Empty response")
        exit(1)

    return response_text

# Set up conversation context
context = "Please make your responses very conversational and polite"
while True:
    print("\nWhat's on your mind? (Type 'exit' to quit)")
    input_str = input(">>> ")

    if input_str == "exit":
        print("Goodbye!")
        exit(0)
    elif input_str == "clear":
        os.system('clear')
        continue

    response = send_query(input_str, context)

    print(response)

    context = f"{context} {input_str} {response}"
    context = context.replace("\n", "")

