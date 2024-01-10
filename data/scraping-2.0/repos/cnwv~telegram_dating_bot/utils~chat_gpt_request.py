import openai
from config import OpenAI, Telegram
import asyncio
from db.commands import db

apy_key_list = eval(OpenAI.api_key)

DEFAULT = 0
ONLINE_STATE = 1
OFFLINE_STATE = 2
RELATIONSHIP_STATE = 3

message_prefix = {
    ONLINE_STATE: 'Знакомство по сети. Подробности:',
    OFFLINE_STATE: 'Знакомство вживую. Подробности:',
    RELATIONSHIP_STATE: 'Взаимоотношения с партнёром. Подробности:'
}
ERROR_TEXT = "Ошибка при выполненнии запроса. Пожалуйста, вернитесь в главное меню"


async def requests_gpt(text, id, username=None, another_choice=False):
    if not db.is_attempt_expire(id):
        return Telegram.expire_text
    api_key = apy_key_list.pop(0)
    openai.api_key = api_key
    apy_key_list.append(api_key)
    if not another_choice:
        state = db.get_message_state(id)
        if state != DEFAULT:
            text = f"{message_prefix[state]} {text}"
            db.set_message_state_to_default(id)
        # отправляем username для ситуаций когда пользователь пропал из бд
        dialog = db.add_message(id, text, "user", username)
    else:
        # если сгенерить другой вариант, то state не нужен
        dialog = db.add_message(id, text, "user", username)
    try:
        print('GPT request')
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=dialog, max_tokens=2000)
        response = completion.choices[0].message.content
    except openai.error.InvalidRequestError:
        print("!!!Error with message size!!!")
        dialog = db.add_message(id, cut_response=True)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=dialog, max_tokens=2000)
        response = completion.choices[0].message.content
    except Exception as e:
        print("!!!Unhandled error: ", e)
        response = ERROR_TEXT
    if response != ERROR_TEXT:
        db.add_message(id, response, 'assistant')
    return response


if __name__ == '__main__':
    async def main():
        # Передача 1-2 сообщений
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"}
        ]

        response = await requests_gpt(messages)
        print("GPT response:", response)


    # Запуск асинхронной функции
    asyncio.run(main())
