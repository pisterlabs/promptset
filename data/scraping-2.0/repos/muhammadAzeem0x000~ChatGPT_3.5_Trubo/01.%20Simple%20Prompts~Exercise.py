from typing import List
import os
import openai
import secret

with open('C:/Users/AZEEM/Desktop/API.txt') as file:
    api_key = file.read().strip()

openai.api_key = api_key

def chatfunctions(prompts: List[str], temp: float, max_t: int) -> List[str]:
    responses = []
    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=max_t
        )
        responses.append(response['choices'][0]['message']['content'])
    return responses

def user_experience(prompt: str, temp: float, max_t: int):
    response = chatfunctions([prompt], temp, max_t)
    print("User: ", prompt)
    print("AI: ", response[0])

user_experience('write a code to in python to capitalize the string', 0.6, 150)