import openai

openai.api_key = "yourChatGPTKey"


def get_response(input: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
             "You are a helpful assistant."},
            {"role": "user", "content": input},
        ],
        temperature=0.5,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["\nUser:"],
    )

    bot_response: str = response["choices"][0]["message"]["content"]
    print("Bot's response:", bot_response)
    return bot_response
