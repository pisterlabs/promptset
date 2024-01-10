import os
import openai
openai.api_key = os.getenv("OPEN_AI_API_KEY")

messages = []
system_msg = "A helpful assistant who gives advice on travelling between 2 points"
messages.append({"role":"system","content": system_msg})

def get_recommendation(prompt):
    messages.append({"role": "system", "content": prompt})
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
    )
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print(f'ChatGPT: {reply}')
    return reply