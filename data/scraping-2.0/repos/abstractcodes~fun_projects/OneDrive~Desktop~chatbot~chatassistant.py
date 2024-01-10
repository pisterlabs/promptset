import openai

openai.api_key = "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"

# appending to the messages.
messages = []
system_msg = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg}) # getting inputs

print("Your new assistant is ready!")
# while loop for talking and resposing to the assistant.
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
# whatever messages you get back as an reply we need to store it and append it to the message.
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")