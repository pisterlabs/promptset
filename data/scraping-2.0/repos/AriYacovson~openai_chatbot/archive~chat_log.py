import dotenv
import openai

# Configure OpenAI API key
openai.api_key = dotenv.get_key(".env", "OPENAI_API_KEY")

chat_log = []

while True:
    user_input = input()
    if user_input == "stop":
        break

    chat_log.append({'role': 'user', 'content': user_input})

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=chat_log,
        temperature=0.6
    )

    bot_response = response['choices'][0]['message']['content']
    chat_log.append({'role': 'assistant', 'content': bot_response})
    print(bot_response)
