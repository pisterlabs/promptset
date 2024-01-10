"""
Elevenlabs api key:
ChatGPT api key
"""

import openai
import os
import json
from elevenlabslib import *

#Load JSON
with open(os.path.join(os.getcwd(), "config.json"), encoding="utf-8") as f:
    config = json.load(f)

# API key
openai.api_key = config["openai"]

def send_message(topic):
    messages = [{"role": "system", "content": "You are a scientist who works on microbial bioinformatics and genomics, and you are a podcast creator."}]
    prompt = f"You would like to generate a podcast episode on {topic}. Could you generate the script for this episode with 1000 words. Don't include music or anything that would be seen as stage directions. Also do not include the person who is speaking, just have the speech. You can include technical detail"
    messages.append({"role": "user", "content": prompt})
    response = []
    response.append(openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10
    ))
    message = str(response[0]["choices"][0]["message"])
    messages.append({"role": "assistant", "content": message})
    messages.append({"role": "user", "content": "Now generate show notes"})
    response.append(
        openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10
    ))

    return response

def main():
    topic = input("What topic would you like this to be on?   ")
    try:
        os.mkdir(os.path.join(os.getcwd(), topic))
    except Exception:
        pass
    text = send_message(topic)
    with open(os.path.join(os.path.join(os.getcwd(), topic), "request1.json"), "w") as f:
        f.write(str(text[0]))
        f.close()
    with open(os.path.join(os.path.join(os.getcwd(), topic), "request2.json"), "w") as f:
        f.write(str(text[1]))
        f.close()
    updatedScript = input(f"{text[0]['choices'][0]['message']}\nWould you like any changes to be made to the script, if so put the updated information.")
    if not updatedScript:
        updatedScript = text[0]['choices'][0]['message']
    
    updatedNotes = input(f"{text[0]['choices'][0]['message']}\nWould you like any changes to be made to the script, if so put the updated information.")
    if not updatedNotes:
        updatedNotes = text[1]['choices'][0]['message']
    
    with open(os.path.join(os.path.join(os.getcwd(), topic), "script.txt"), "w") as f:
        f.write(f"Show notes for the podcast:\n\n{updatedNotes}\n\nScript for the podcast:\n\n{updatedScript}")
    
    

if __name__ == "__main__":
    main()