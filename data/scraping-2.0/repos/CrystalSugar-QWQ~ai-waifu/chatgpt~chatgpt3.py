import openai
import json
import sys
import time

import chatgpt
from config import api_key


# to help the CLI write unicode characters to the terminal
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

conversation = []
history = {"history": conversation}

openai.api_key = api_key
outputNum = 20  # 限制回答长度
conversation_file = "./conversation.json"    # 历史文件地址


# function to get an answer from OpenAI
def openai_answer():
    global conversation

    total_characters = sum(len(d['content']) for d in conversation)

    while total_characters > 4000:
        try:
            # print(total_characters)
            # print(len(conversation))
            conversation.pop(2)
            total_characters = sum(len(d['content']) for d in conversation)
        except Exception as e:
            print("Error removing old messages: {0}".format(e))

    with open(conversation_file, "w", encoding="utf-8") as f:
        # Write the message data to the file in JSON format
        json.dump(history, f, indent=4)

    prompt = chatgpt.getPrompt()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=200,
        temperature=1,
        top_p=0.9
    )
    message = response['choices'][0]['message']['content']
    conversation.append({'role': 'assistant', 'content': message})

    # translate_text(message)
    return message


# 使用openai进行文字提取(记得去填你的api)
def transcribe_audio(file):
    audio_file = open(file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    chat_now = transcript.text
    return chat_now
    

def chatgpt_answer(name, text):
    global conversation
    # result = name + " said " + text
    result = text
    conversation.append({'role': 'user', 'content': result})
    while True:
        try:
            message = openai_answer()
            break
        except Exception as e:
            print("[openai error]:", e)
            print("[等待20s再次询问]")
            time.sleep(10.1)
            print("[剩余10s]", end="", flush=True)
            time.sleep(5)
            print("[剩余5s]", end="", flush=True)
            time.sleep(2)
            print("[剩余3s]", end="", flush=True)
            time.sleep(1)
            print("[剩余2s]", end="", flush=True)
            time.sleep(1)
            print("[剩余1s]")
            time.sleep(1)
    # message = openai_answer()
    return message
