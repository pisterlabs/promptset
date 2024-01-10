import openai
import os
 
#API key
openai.api_key = "YOUR API KEY"
 
def chatbot():
  # Creeare lista stocare mesaje
  messages = [
    {"role": "system", "content": "You are a sarcastic assistant."},
  ]
 
  while True:
    message = input("User: ")
 
    if message.lower() == "quit":
      break
 
    messages.append({"role": "user", "content": message})
 
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
 
    chat_message = response['choices'][0]['message']['content']
    print(f"NAONbot: {chat_message}")
    messages.append({"role": "assistant", "content": chat_message})
 
if __name__ == "__main__":
  print("Start chatting with the bot (type 'quit' to stop)!")
  chatbot()
