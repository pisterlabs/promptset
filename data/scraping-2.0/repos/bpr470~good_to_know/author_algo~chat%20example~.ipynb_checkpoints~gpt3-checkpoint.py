import os
import openai

openai.api_key = "sk-u3p3gFAO8YK4sCt629Ido8aG35OWSB8Ovvic3bfB"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

def response(a):
    response = openai.Completion.create(
      engine="davinci",
      prompt="\n\nHuman:"+a+"AI:",
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      presence_penalty=0.6,
      stop=["\n", " Human:", " AI:"]
    )
    return response['choices'][0]['text']