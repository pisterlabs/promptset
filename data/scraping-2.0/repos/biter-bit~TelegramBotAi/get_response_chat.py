import openai, os, dotenv, json, sys

# Загрузка переменных окружения, включая API-ключ OpenAI
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def send_message_and_update_history(messages):
    # Отправляем запрос с текущей историей сообщений
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Получаем ответ бота
    bot_message = response.choices[0].message.content

    return bot_message


def load_messages_file(id):
    if os.path.isfile(f'messages/{id}.json'):
        with open(f'messages/{id}.json', 'r', encoding='utf-8') as file:
            content = file.read()
            if content:
                messages = json.loads(content)
            else:
                messages = {}
    else:
        with open(f'messages/{id}.json', 'w') as file:
            pass
        messages = {}
    return messages


def save_messages_user(id, user_messages):
    with open(f"messages/{id}.json", "r") as file:
        content = file.read()
    with open(f"messages/{id}.json", "w") as file:
        if content:
            messages = json.loads(content)
        else:
            messages = {}
        if str(id) in messages:
            messages[str(id)] = messages[str(id)] + user_messages
        else:
            messages[str(id)] = user_messages
        json.dump(messages, file, indent=6)