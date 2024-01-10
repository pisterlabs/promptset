import json
import time

import openai
from ChatGPT import BOT_KEY


openai.api_key = BOT_KEY

intents = json.loads(open('intents.json').read())

messages = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        message = pattern
        messages.append({"role": "user", "content": "35 words" + message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = messages
        )
        reply = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        if reply not in intent['responses']:
            intent['response'].append(reply)
        time.sleep(90)
        with open('intents.json', 'w') as file:
            json.dump(intents, file, indent=4)
        