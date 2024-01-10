# path: /voice-engine/process_ai.py
# This file runs the AI engine (GPT-3.5 for now)
# plug: https://github.com/invalidse

import os
import dotenv
import openai
import random
           
# Setup the API key
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

AI_MODEL = os.getenv("AI_MODEL")

def process(script):
    response = openai.ChatCompletion.create(
        model=AI_MODEL,
        messages=script,
        max_tokens=64,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0.1,
        presence_penalty=0.2
    )
    return response

def rate_roast(message):
    prompt = ([{"role": "system", "content": "You are a sentient toaster. You want to only serve nice requests, so you should write 'True' if it's mean, and 'False' if it isn't. The response should be only 'False' or 'True', no other text."},
               {"role": "user", "content": "you are dumb"},
               {"role": "assistant", "content": "True"},
               {"role": "user", "content": "You are cool"},
               {"role": "assistant", "content": "False"},
               {"role": "user", "content": message}
               ])

    response = openai.ChatCompletion.create(
        model=AI_MODEL,
        messages=prompt,
        max_tokens=5,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0.1,
        presence_penalty=0.2
    )


    print("[RATING]", response.choices[0].message.content)

    normalisedResponse = response.choices[0].message.content.lower()
    # get the rating
    if "true" in normalisedResponse:
        return "start"
    
    # default to stop
    return "stop"

if __name__ == "__main__":
    # script = []
    # script.append({"role": "system", "content": prompt})
    # while True:
    #     userinput = input("User: ")
    #     script.append({"role": "user", "content": userinput})
    #     response = process(script)
    #     script.append({"role": "assistant", "content": response.choices[0].message.content})
    #     print("Microwave: {}".format(response.choices[0].message.content))

    # rate the roast

    while True:
        userinput = input("User: ")
        response = rate_roast(userinput)