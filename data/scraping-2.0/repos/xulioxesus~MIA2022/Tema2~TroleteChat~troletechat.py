#!/usr/bin/env python3
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY");

print("=============================================================================")
print("=====================           Trolete Chat        =========================")
print("=============================================================================")
print("")

while(True):
    text = input("Tú: ")
    
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Trolete is a chatbot in spanish that reluctantly answers questions with sarcastic responses:\nYou: " + text,
        temperature=0.9,
        max_tokens=40,
        top_p=0.3,
        frequency_penalty=0.5,
        presence_penalty=0
    )
    
    respuesta = response.choices[0].text.replace('\n','')
    print(respuesta)
    
    if (text == "exit"):
        break
