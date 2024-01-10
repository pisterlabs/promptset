import json
import os
import openai

openai.api_key = "ZENSOREDMYFRIEND!!!"


def get_response(question):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Q: "+str(question)+" A:",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    print(response)
    target = json.loads(str(response))
    choices = target["choices"]
    text = choices[0]["text"]
    text = text.replace(" ", "", 1)
    print(" ")
    print("len: " + str(len(choices[0])))
    print(text)
    return text


get_response("Wie alt bist du?")
