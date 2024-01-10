#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openai
import os
import sys
from boto3 import client

# OpenAI API key setup
api_key = "sk-"
openai.api_key = api_key

# FAQ data
faq_data = [
    {
        "question": "What is your return policy?",
        "answer": "Our return policy lasts 30 days from the date of purchase..."
    },
    {
        "question": "How do I track my order?",
        "answer": "Once your order is shipped, you will receive a tracking link..."
    },
    # Add more FAQ questions and answers here.
]

def get_gpt_answer(question):
    # Use GPT-3.5 API to generate an answer for all questions
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Main loop to receive user input and provide answers
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Check if the user's question matches any FAQ question
    faq_answer = next((faq["answer"] for faq in faq_data if faq["question"].lower() == user_input.lower()), None)

    if faq_answer:
        print("FAQ Bot:", faq_answer)
    else:
        # If the question is not in the FAQ data, use GPT-3.5 for the answer
        gpt_answer = get_gpt_answer(user_input)
        print("GPT-3.5:", gpt_answer)

    polly = client("polly", region_name="ap-northeast-2")
    response = polly.synthesize_speech(
            Text=gpt_answer,
            OutputFormat="mp3",
            VoiceId="Seoyeon")

    stream = response.get("AudioStream")

    with open('aws_test_tts.mp3', 'wb') as f:
        data = stream.read()
        f.write(data)


    cmd = "omxplayer -o both aws_test_tts.mp3"

    return_value = os.system(cmd)
    print('return value: ', return_value)
