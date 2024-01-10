import openai
import time
import os


openai.api_key = "sk-"

messages = []
system_msg = input("What type of Chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})

print("Your new assistant is ready")

while True:
    user_input = input("")
    if user_input == "quit()":
        break
    messages.append({"role": "user", "content": user_input})

    # Call the OpenAI API to get a response
    response = openai.Completion.create(
        engine="davinci",
        prompt="\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]),
        temperature=0.7,
        max_tokens=50
    )

    # Append the model's response to the messages
    model_response = response.choices[0].text.strip()
    messages.append({"role": "assistant", "content": model_response})

    print("Assistant:", model_response)
    
