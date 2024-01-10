import os
import openai

openai.api_key = "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"

print("Bienvenido al ChatBot. \n "
      "Cuando te canses de hablar con una máquina y decidas buscar un amigo de verdad, solo "
      "escribe 'Adios.', y automáticamente terminará la conversación. \n")

conversation = ""
question = 1

while question != "Adios.":
    question = input("Human: ")
    conversation += "\nHuman: " + question + "\nAI:"
    response = openai.Completion.create(
        engine="davinci",
        prompt=conversation,
        temperature=0.75,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=5.0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    answer = response.choices[0].text.strip()
    conversation += answer
    print("AI: " + answer + "\n ")
