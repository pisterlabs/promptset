#! /usr/bin/env python

import os
import openai, config

init_message = [{'role':'system', 'content':'You are a helpful AI assistant named Laozibot.'}]
openai.api_key = config.OPENAI_API_KEY

chat = init_message.copy()

def get_lines():
    filename = "history.txt"
    last_10_messages = []

    with open(filename, "r") as file:
        lines = file.readlines()
        i = len(lines) - 1
        while i >= 0 and len(last_10_messages) < 10:
            line = lines[i].strip()
            if line.startswith("User:") or line.startswith("Assistant:"):
                message = line
                i -= 1
                while i >= 0 and not lines[i].strip().startswith(("User:", "Assistant:")):
                    message = lines[i].strip() + "\n" + message
                    i -= 1
                last_10_messages.append(message)
            else:
                i -= 1

    last_10_messages.reverse()

    return last_10_messages

def summarize(chat_hist):
    print("Summarizing last 10 chats")
    prompt = [{'role':'system', 'content':'You are a summary bot. The user input is a chat transcript between a user and an assistant named Laozibot. Respond to each message with a concise bulleted list of the facts about both the user and the assistant.'}]
    chat = "\n".join(chat_hist) 
    prompt.append({'role':'user', 'content':chat})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt)
    m = response["choices"][0]["message"]
    print(m['content'])
    new_prompt = [{'role':'system', 'content':"You are a helpful AI assistant named Laozibot. You have a long-term memory summarized as a list of facts here: {}\nuse this information to inform all future responses".format(m['content'])}]
    return new_prompt



u = input("Laozi: ")
while True:
    chat.append({"role":"user", "content":u})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=chat)
    m = response["choices"][0]["message"]
    sys_message = m['content']
    
    print("Laozibot: {}".format(sys_message))
    chat.append(m)
    with open("history.txt", 'a') as f:
        f.write("User: {}\n Assistant: {}\n".format(u, sys_message))
        f.close()
    
    chat_hist = get_lines()
    #if len(chat_hist) % 2 == 0:
    chat = summarize(chat_hist)
    
    u = input("Laozi: ")

        
