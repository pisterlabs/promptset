import openai

bot_api_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

while True:
    user_input = input("You: ")
    bot_api_messages.append({"role": "user", "content": user_input})
    bot_response = openai.ChatCompletion.create(model="gpt-4", messages=bot_api_messages)
    bot_response_content = bot_response["choices"][0]["message"]["content"]
    bot_api_messages.append({"role": "assistant", "content": bot_response_content})
    print("\nBot: ", f"{bot_response_content}\n")