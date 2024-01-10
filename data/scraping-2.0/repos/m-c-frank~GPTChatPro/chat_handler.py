import openai
import os
import dotenv

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_answer(messages):
  print(messages)
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )
  return response["choices"][0]["message"]

if __name__ == "__main__":
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        assistant_response = get_answer(messages)
        messages.append(assistant_response)
        response_text = assistant_response["content"]
        print("Assistant:", response_text)  