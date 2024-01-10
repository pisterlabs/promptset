#!/usr/bin/env python
# coding = utf8
#
# Copyright (C) 2023, Diego Cambiaso
# GNU General Public License v3.0

'''
This script is for educational purposes and will let you run a simple Chat GPT implementation.
'''

# (*) You need to create a free account to get your API Key.
# openai.com

import openai

openai.api_key = (
    "XX-XXPfEU2qNPP5s3kPOoyxT3BlbkFJdU7r7mfV0KX39ZQ9KCXX"  # please, change the API KEY (*)
)

print("\nWelcome to Chat AI")
print("This chat is OpenAI ChatGPT based")

while True:
    try:
        question = input("\nWrite a question: ")

        if question.upper() == "EXIT":
            print("Good bye")
            exit()

        completion = openai.Completion.create(
            engine="text-davinci-003", prompt=question, max_tokens=2048
        )

        print(completion.choices[0].text)  # type: ignore

    except openai.error.RateLimitError as APIError:
        print(APIError)
        print("Rate limit exced")
    except FileNotFoundError:
        print("Error during input data or OpenAI API is not available at this moment")
        print('Please try again or write "Exit"')
        continue
    except KeyboardInterrupt:
        print("\n\nFinished by the user - Good bye!")
        exit()
