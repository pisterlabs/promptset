import os
from openai import OpenAI

messages = [ {"role": "system", "content":
            "You are a intelligent assistant."} ]
# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI()

while True:
    print("waiting user input ...")
    message = input("User : ")
    print("user input \"%s\"" % message) 
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        print("before completions.create: %d" % len(messages))
        for m in messages:
            print(m)
        completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages)
        print("after completions.create")
        reply = completion.choices[0].message.content
        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})
