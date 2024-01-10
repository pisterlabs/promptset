import os
import openai
openai.api_key = "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"

chat_messages=[]

while True:
    user_input=input("You: ")

    chat_messages.append({"role":"user","content":user_input})

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=chat_messages,
        temperature=0
    )


    ai_response = response.choices[0].message.content
    chat_messages.append({"role":"user","content":ai_response})

    print("AI: ", ai_response)
    print(chat_messages)