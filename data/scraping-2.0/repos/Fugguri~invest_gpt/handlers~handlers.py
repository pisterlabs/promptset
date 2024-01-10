import json
from main import OpenAI, OPENAI_KEY, config
import httpx
import asyncio
from aiogram import types, Bot
from main import dp, bot, db, channels, js
from channel_joined import get_channel_member, is_member_in_channel
from openai import RateLimitError, BadRequestError
users_message = {}

proxy = config["proxy"]
openai = OpenAI(
    api_key=OPENAI_KEY,

    http_client=httpx.Client(
        proxies=proxy,
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):

    await message.answer("""Привет!

Вы попали в бот со встроенным искусственным интеллектом, данный бот поможет вам привлечь инвестиции в ваш проект!   

Вы можете общаться с ботом, как с живым собеседником, задавая вопросы на любом языке.

🚀 Помните, что ботом вместе с вами пользуются ещё тысячи человек, он может отвечать с задержкой.""")
    db.add_user(full_name=message.from_user.full_name,
                telegram_id=message.from_user.id,
                username=message.from_user.username
                )


@dp.message_handler(commands=['add_chat'])
async def mailing(message: types.Message):
    result = message.get_args()
    result = list(
        map(lambda x: x.strip().replace("\n", ""), result.split(" ")))
    group_id = str(result[0])
    group_name = result[1]
    await message.bot.get_chat_member(group_id, message.from_user.id)
    try:
        member = await get_channel_member(group_id, message)
    except:
        await message.answer("Ошибка, проверьте есть ли у бота права администратора")
        return
    if js.set_new_channel_for_subscribe(group_id, group_name):
        await message.answer("Успешно")
    else:
        await message.answer("Ошибка")


@dp.message_handler(commands=['delete_chat'])
async def mailing(message: types.Message):
    result = message.get_args().split(" ")
    group_id = str(result[0]).strip()
    if js.delete_channel_for_subscribe(group_id):
        await message.answer("Успешно")
    else:
        await message.answer("Ошибка")


@dp.message_handler(commands=['mailing'])
async def mailing(message: types.Message):
    result = message.get_args()
    users = db.get_all_users()
    for user in users:
        await bot.send_message(chat_id=user[2], text=result)


@dp.callback_query_handler(lambda call: call.data == "check")
async def check(callback: types.CallbackQuery):
    channels_text = ""
    all_joined = True
    for key, value in js.get_channels().items():
        member = await get_channel_member(key, callback)
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton(
            text="Проверить подписку", callback_data="check"))
        if not is_member_in_channel(member):
            all_joined = False
            channels_text += "\n"+value

    if not all_joined:
        await callback.message.answer(text=f"Нет подписки.\nЧтобы пользоваться сервисом - подпишитесь {channels_text}",
                                      reply_markup=markup)
    else:
        await callback.message.answer("Поздравляем, теперь вы можете пользоваться ботом.\nЧтобы начать пользоваться напишите вопрос или любое сообщение.")


@dp.message_handler()
async def communicate(message: types.Message):
    wait = await message.answer("Форматируется ответ…")
    try:
        try:
            users_message[message.from_user.id]
        except:
            users_message[message.from_user.id] = []
        users_message[message.from_user.id].append(
            {"role": "user", "content": message.text})
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=users_message[message.from_user.id]
        )
        answer = response.choices[0].message.content
        users_message[message.from_user.id].append(
            {"role": "assistant", "content": answer})
        await message.reply(answer)
    except RateLimitError as ex:
        print(ex)
        await asyncio.sleep(20)
        await wait.delete()
        await communicate(message)
    except BadRequestError as ex:
        print(ex)
        users_message[message.from_user.id] = []
        await wait.delete()
        await communicate(message)

    except Exception as ex:
        print(ex)
        users_message[message.from_user.id] = []
        await wait.delete()
        await communicate(message=message)
        # await message.reply("Не понимаю, сформулируйте подругому")
