import openai

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a CBT therapist"}]


def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply