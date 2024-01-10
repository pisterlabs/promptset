from dotenv import load_dotenv
import os
from openai import OpenAI
import json

from termcolor import colored

import google.generativeai as genai



# This method will load the .env file variables to the environment variables
load_dotenv(override=True)



client = OpenAI()

# models = client.models.list().model_dump()
# model_ids = [item['id'] for item in models['data']]
# for model_id in model_ids:
#     print(model_id)

history = [
    {"role": "system", "content": "You love informal debate and specialize in witty retorts and try to cut the opponents positon to ribbons. Give exctly one argument or rebuttal at a time, no list, but questions (retorhical or otherwise) are good. Avoid repeating yourself, instead find a new angle of attack. Keep responses short as possible. Do not summarize, just state your argument and await a response. Today you are taking the position that God does exist."},
]


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

config = genai.types.GenerationConfig(max_output_tokens=512)

print("Debate is about to begin...")
print()

try:
    gresponse = chat.send_message(
        "You are a skilled debater in the mold of Christopher Hitchens, and love to descimate your opponent with brilliant insights. Give exctly one argument or rebuttal at a time, no list, but questions (retorhical or otherwise) are good. Avoid repeating yourself, instead find a new angle of attack. Today you are taking the position that God does not exist.", generation_config=config)
except Exception as e:
    print(e)
    print("Something went wrong. Please try again.")
    exit()

while True:
    print()
    print(colored("GEMINI:",'yellow'), colored(gresponse.text,'light_green'))
    history.append({"role": "user", "content": gresponse.text})

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=history
    )

    assistant_output = completion.choices[0].message.content
    print()
    print(colored("ChatGPT:",'cyan'),colored(assistant_output,'light_blue'))

    history.append({"role": "assistant", "content": assistant_output})
    
    gresponse = chat.send_message(assistant_output, generation_config=config)
    
