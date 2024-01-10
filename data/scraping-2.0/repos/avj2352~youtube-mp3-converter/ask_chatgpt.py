'''
Simple ChatGPT interface
getting familiar with openai
https://github.com/openai/openai-quickstart-python/blob/master/app.py
'''

import openai
from util.config import OPEN_AI_API_KEY, typing_print

openai.api_key = OPEN_AI_API_KEY
model_engine = 'text-davinci-003'

def chat_gpt(text: str):
    print('generating response...')
    response = openai.Completion.create(
            engine=model_engine,
            prompt=text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5
            )
    typing_print(response.choices[0].text)

# print(OPEN_AI_API_KEY)

# Keep prompting until "thanks" is entered
while True:
    try:        
        prompt: str = input('\n🤖 Ask chatgpt: ')
        if prompt == "" or prompt == "thanks" or prompt == "quit" or prompt == "q":
            break
        else:
            chat_gpt(prompt)
            continue
    except ValueError:
        print("🤖 Sorry, I didn't understand that!")
        continue

print("🤖 Nice chatting!")
