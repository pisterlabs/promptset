"""
Some premade bots
"""
import os
import pickle
from typing import cast

from openai.types.beta import Thread

from chats.bot_shell import Bot, BotConversation


async def get_persistent_bot(bot_name: str, bot_instructions: str, model: str):
    bot_pickle_file_name = bot_name.replace(" ", "_").lower()
    file_path = os.path.join(os.path.dirname(__file__), f"{bot_pickle_file_name}.pkl")

    if os.path.exists(file_path):
        with open(file_path, "rb") as bot_file:
            unpickled_bot_data = pickle.load(bot_file)
            name_bot = Bot(assistant_id=unpickled_bot_data.id, model=model)
            await name_bot.populate_assistant()
    else:
        name_bot = Bot(model=model)
        await name_bot.create_assistant(bot_name, bot_instructions)
        with open(file_path, "wb") as bot_file:
            pickle.dump(name_bot.assistant, bot_file)

    return name_bot


def calculate_thread_file_name(bot_name: str, thread_id: str):
    if not thread_id:
        raise ValueError("thread_id must be specified")
    bot_name = bot_name.replace(" ", "_").lower()
    file_path = os.path.join(os.path.dirname(__file__), f"{bot_name}/{thread_id}.pkl")
    return file_path


async def get_persistent_bot_convo(bot: Bot, thread_id: str) -> BotConversation:
    # make folder with name of bot if it doesn't exist
    safe_bot_name = bot.assistant.name.replace(" ", "_").lower()
    if not os.path.exists(safe_bot_name):
        os.makedirs(safe_bot_name)

    if thread_id:
        file_path = calculate_thread_file_name(bot.assistant.name, thread_id)
        with open(file_path, "rb") as convo_file:
            unpickled_thread = cast(Thread, pickle.load(convo_file))
            convo = BotConversation(bot.assistant, unpickled_thread)
            await convo.populate_thread()
            return convo

    convo = BotConversation(bot.assistant)
    await convo.create_thread()
    file_path = calculate_thread_file_name(bot.assistant.name, convo.thread_id)
    with open(file_path, "wb") as convo_file:
        pickle.dump(convo.thread, convo_file)

    return convo
