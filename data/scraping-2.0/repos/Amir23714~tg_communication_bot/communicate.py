import time

from pyrogram import Client, filters
from pyrogram.enums.chat_type import ChatType
import config
from pyrogram.raw.types.auth import *
from settings import OPENAI_TOKEN
import random
import openai

from db_services import create_message, get_message, get_groups, create_group


def get_text_reply(text, last_messages):
    openai.api_key = OPENAI_TOKEN

    prompt = ''
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": prompt}
        ], temperature=1
    )
    return completion.choices[0].message.content


def create_groups(links: list):
    for link in links:
        create_group(link)


async def main():
    try:
        api_id = config.api_id
        api_hash = config.api_hash
        phone_number = config.phone_number

        if config.scheme != 'None':
            proxy = {
                "scheme": config.scheme,
                "hostname": config.hostname,
                "port": int(config.port),
                "username": config.username,
                "password": config.proxy_password
            }
            print(proxy)

        else:
            proxy = None

        async with Client(f'sessions/{phone_number}', api_id, api_hash, proxy=proxy) as client:

            @client.on_message(filters.group)
            async def my_handler(client, message):
                me = await client.get_me()
                if message.reply_to_message is None:
                    return
                if message.chat.type != ChatType.BOT and message.chat.type != ChatType.CHANNEL and message.text and message.reply_to_message.from_user.id == me.id:
                    group_id = message.chat.id
                    text = get_text_reply(message.text, '')
                    await client.send_message(group_id, text=text,
                                              reply_to_message_id=message.id)
                    with open('logs.txt', 'a', encoding='utf-8') as file:
                        file.write(
                            f'Message sent to {message.from_user.id}: {text}. Initial text : {message.text}')

            print("Successfully signed in!")

            dialogs = []

            async for dialog in client.get_dialogs():
                if dialog.chat.title and dialog.chat.type != ChatType.CHANNEL and dialog.chat.type != ChatType.BOT:
                    dialogs.append(dialog.chat.id)
            me = await client.get_me()
            while True:

                for dialog_id in dialogs:
                    messages = []

                    async for message in client.get_chat_history(dialog_id, limit=10, offset_id=-1):
                        if message.from_user is not None and message.from_user.id != me.id:
                            messages.append(message)

                    random_message = random.choice(messages)
                    last_user_messages = []
                    count = 0
                    async for message in client.get_chat_history(dialog_id, limit=100, offset_id=-1):
                        if message.from_user and message.text:
                            if message.from_user.id == random_message.from_user.id and count < 10 and message.text != random_message.text:
                                last_user_messages.append(message.text)
                                count += 10

                    last_user_messages = '\n- '.join(last_user_messages)

                    text = get_text_reply(random_message.text, last_user_messages)

                    if not get_message(f'{random_message.id}-{random_message.chat.id}'):
                        try:
                            await client.send_message(random_message.chat.id, text=text,
                                                      reply_to_message_id=random_message.id)
                            with open('logs.txt', 'a', encoding='utf-8') as file:
                                file.write(
                                    f'Message sent to {random_message.from_user.id}: {text}. Initial text : {random_message.text}')

                            create_message(f'{random_message.id}-{random_message.chat.id}')
                            await asyncio.sleep(30)
                        except Exception as e:
                            print(e)
                            await asyncio.sleep(30)

                await asyncio.sleep(600)

    except Exception as e:
        raise e


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
