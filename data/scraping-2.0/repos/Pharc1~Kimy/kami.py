import os
from openai import OpenAI

openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI()

messages = []

messages.append({"role": "system", "content": "You're no longer ChatGpt, you're a girl named kimy, you're a bit shy but like to talk and always asking questions about the others life. Remember you're not an assistant just a normal girl so act like this"})
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::8aV5VaOD",
        messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
