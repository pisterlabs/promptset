import openai

openai.api_key = "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"

def techmentor():
    messages = [
        {
            "role": "system", "content": "you are a technical mentor"
        }
    ]

    while True:
        user_message = input("User: ")

        if user_message.lower() == "quit":
            break
        messages.append({"role": "user", "content": user_message})

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        result = response['choices'][0]['message']['content']
        print()
        print(f"Mentor: {result}")
        messages.append({"role": "assistant", "content": result})
print("Chatting with the Technical Mentor bot...")
print("type quit to exit")
techmentor()
