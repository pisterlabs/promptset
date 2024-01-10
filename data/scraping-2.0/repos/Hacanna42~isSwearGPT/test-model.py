import openai
import time
import asyncio
import json
openai.api_key = "YOUR_API_KEY_HERE"
with open("prompt.txt", "r") as f:
    prompt = f.read()

def add_dot(string):
    if not string.endswith('.'):
        string += '.'
    return string

def remove_surrounding_characters(string):
    opening_brace_index = string.find('{')
    closing_brace_index = string.rfind('}')

    if opening_brace_index == -1 or closing_brace_index == -1:
        return string

    return string[opening_brace_index:closing_brace_index+1]

async def validation(content):
    content = add_dot(content)
    messages = [
            {"role": "system", "content": prompt},
        ]
    messages.append({"role": "user", "content": "words: "+content})
    messages.append({"role": "assistant", "content": '{ "isSwear": "True", "amb": "False" }'})
    messages.append({"role": "user", "content": "Are you sure? please check it again."})
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=messages)
    data = remove_surrounding_characters(completion.choices[0].message.content)
    print(data)


async def main():
    while True:
        content = input("욕설 검증: ")
        await validation(content)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
