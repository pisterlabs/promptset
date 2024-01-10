import openai
from apikey import api

openai.api_key = api
messages=[
    {
      "role": "system",
      "content": """This is where you write the prompts"""
    }
  ]

while input != "quit()":
  message = input("")
  #for i in messages:
  #  print(f"Role: {i['role']}, Content: {i['content']}")
  messages.append({"role":"user","content":message})
  response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages = messages,
  temperature=1,
  max_tokens=1024,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
  reply = response["choices"][0]["message"]["content"]
  messages.append({"role":"assistant","content":reply})
  print("\n"+reply+"\n")