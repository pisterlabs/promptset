#! /usr/bin/env python3

from openai import OpenAI
client = OpenAI()

question = input("Ask your question: ")
m = [{"role": "system", "content": "You are a code helper. You respond to each message with only the relevant code. If there is no relevant code in the response, explain the response with the smallest amount of text possible."}]

while question != "quit":
    m.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
      model="gpt-4",
      messages=m
    )
    m.append({"role": completion.choices[0].message.role, "content":completion.choices[0].message.content})
    print("Laozi_shell: ", completion.choices[0].message.content)
    
    question = input("You: ")


