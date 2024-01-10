import os
from flask import Flask, request, jsonify
import openai
import constants
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# export OPENAI_API_KEY="YOUR_API_KEY"
# export SD_API_KEY="YOUR_API_KEY"

# PURPOSE OF THIS TEST
# This is a test to see what instruction would give the best result in converting a
# summary into a sentence with the best result

# How to run?
# in your terminal, run the following command:
# python ./art_test.py
# Make sure that you are in the correct relative directory
# A window should pop up with the generated art pieces

openai.api_key = os.getenv("OPENAI_API_KEY")

instructions = constants.instructions_sentence[0]
summaries = constants.summaries

with open('sentence_test.txt', 'a') as file:
    count = 0
    for summary in summaries:
        try:
            count += 1
            print("Starting sentence eneration for instruction " + str(count))
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": summary}
                ]
            )
            print("Completed sentence generation generation")
            sentence = completion.choices[0].message.content.replace("\n", " ")
            print("Sentence: " + sentence)
            file.write("Sentence: " + sentence + "\n\n")
        except Exception as e:
            print(e)
            continue
