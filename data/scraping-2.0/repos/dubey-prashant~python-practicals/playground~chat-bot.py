from openai import OpenAI
from config import OPENAI_API_KEY # local file with secrets 

client = OpenAI(api_key=OPENAI_API_KEY) 

messages=[{"role": "system", "content": "You are a helpful assistant."}]

def showPrompt():
  prompt=input("You: ")
  if prompt == "stop":
    print("Goodbye!")
    return 
  else: 
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    print("Bot: ", completion.choices[0].message.content)
    messages.append( completion.choices[0].message )
    showPrompt()

showPrompt()
