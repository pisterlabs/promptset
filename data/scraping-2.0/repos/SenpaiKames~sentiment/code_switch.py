import openai


openai.api_key = "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"

messages = []
messages.append({"role": "system", "content": "You are a helpful assistant."})

print("Your assistant is ready!")
while input != "quit()":
    message = input("Input: ")
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")




