import logging
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import ParseMode
from aiogram.utils import executor
import openai
import lib.duckduckgo as DDG
import lib.utils

# Load environment variables
params = lib.utils.Parametrize()
params.read_environment()

# Initialize chat history
chat = lib.utils.ChatUtils(params.chat_history_size)

openai.api_key = params.openai_api_key

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=params.telegram_api_token)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


async def is_allowed(user_id: int) -> bool:
    """
    Check if the user is allowed to use the bot.

    :param user_id: User ID

    :return: True if the user is allowed, False otherwise
    """
    if user_id in params.allowed_chat_ids:
        return True
    return params.is_public


async def send_typing_indicator(chat_id: int):
    """
    Send typing indicator to the chat.

    :param chat_id: Chat ID

    :return: None
    """
    await bot.send_chat_action(chat_id, action="typing")


# Command handlers
async def on_start(message: types.Message):
    """
    Send a welcome message when the bot is started.

    :param message: Message object

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    await message.answer(f"Hello! I am a ChatGPT bot.\nI am using {params.gpt_chat_model}.\nType your message and I'll respond.")


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    """
    Send a welcome message when the bot is started.

    :param message: Message object with the /start command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    await message.answer(f"Hello! I'm a ChatGPT bot.\nI am using {params.gpt_chat_model}.\nSend me a message or a command, and I'll respond!")


@dp.message_handler(commands=['help'])
async def help_command(message: types.Message):
    """
    Send a help message with a list of available commands.

    :param message: Message object with the /help command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    help_text = (
        "Here's a list of available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/info - Get information about the bot\n"
        "/status - Check the bot's status\n"
        "/newtopic - Clear ChatGPT conversation history\n"
        "/regen - Regenerate last GPT response\n"
        "/web <query> - Search with Duckduckgo and process results with ChatGPT using query\n"
    )
    await message.answer(help_text)


@dp.message_handler(commands=['info'])
async def info_command(message: types.Message):
    """
    Send a message with information about the bot.

    :param message: Message object with the /info command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    info_text = f"I'm a ChatGPT bot.\nI am using {params.gpt_chat_model}.\nI can respond to your messages and a few basic commands.\nVersion: {params.version}"
    await message.answer(info_text)


@dp.message_handler(commands=['status'])
async def status_command(message: types.Message):
    """
    Send a message with the bot's status.


    :param message: Message object with the /status command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    status_text = "I'm currently up and running!"
    await message.answer(status_text)


@dp.message_handler(commands=['newtopic'])
async def newtopic_command(message: types.Message):
    """
    Clear ChatGPT conversation history.

    :param message: Message object with the /newtopic command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    chat.clean_chat_history(message.chat.id)
    status_text = "ChatGPT conversation history is cleared!"
    await message.answer(status_text)


@dp.message_handler(commands=['regen'])
async def regenerate_command(message: types.Message):
    """
    Regenerate last ChatGPT response.

    :param message: Message object with the /regen command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    if chat.history.get(message.chat.id):
        chat.remove_last_chat_message(message.chat.id)
    await send_typing_indicator(message.chat.id)
    regen_message = await message.answer("Generating new answer…")
    response_text = await chatgpt_chat_completion_request(chat.history[message.chat.id])
    await regen_message.delete()
    await message.answer(f"Generating new respose on your query:\n<i><b>{chat.history[message.chat.id][-1]['content']}</b></i>\n\n{response_text}", parse_mode=ParseMode.HTML)
    chat.add_chat_message(message.chat.id, {"role": "assistant", "content": response_text})


@dp.message_handler(commands=['web'])
async def websearch_command(message: types.Message):
    """
    Search with Duckduckgo and process results with ChatGPT using query.

    :param message: Message object with the /web command

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return
    query = message.get_args()
    search_message = await message.answer("Searching…")
    await send_typing_indicator(message.chat.id)
    web_search_result, result_status = await DDG.web_search(query, params.num_search_results)
    if result_status == "OK":
        chat_gpt_instruction = 'Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [number] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate anwers for each subject. In the end of answer provide a list of all used URLs.'
        input_text = web_search_result + "\n\n" + chat_gpt_instruction + "\n\n" + "Query: " + query + "\n"
        chat.add_chat_message(message.chat.id, {"role": "user", "content": input_text})
        await send_typing_indicator(message.chat.id)
        response_text = await chatgpt_chat_completion_request(chat.history[message.chat.id])
    if result_status == "ERROR":
        response_text = "No results found for query: " + query + "\n"
    await search_message.delete()
    await message.reply(response_text)
    chat.add_chat_message(message.chat.id, {"role": "assistant", "content": response_text})


@ dp.message_handler(content_types=types.ContentTypes.NEW_CHAT_MEMBERS)
async def new_chat_member_handler(message: types.Message):
    """
    Send a welcome message when the bot is added to a group chat.

    :param message: Message object with the new chat members

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    new_members = message.new_chat_members
    me = await bot.get_me()

    if me in new_members:
        await on_start(message)
        await help_command(message)

# Message handlers


@ dp.message_handler()
async def reply(message: types.Message):
    """
    Reply to a message using ChatGPT.

    :param message: Message object

    :return: None
    """
    if not await is_allowed(message.from_user.id):
        return  # Ignore the message if the user is not allowed
    input_text = message.text
    chat.add_chat_message(message.chat.id, {"role": "user", "content": input_text})
    await send_typing_indicator(message.chat.id)
    response_text = await chatgpt_chat_completion_request(chat.history[message.chat.id])
    await message.reply(response_text)
    chat.add_chat_message(message.chat.id, {"role": "assistant", "content": response_text})


async def chatgpt_chat_completion_request(messages_history):
    """
    Send a request to the ChatGPT API.

    :param messages_history: Chat history

    :return: Response from the ChatGPT API
    """
    try:
        response = openai.ChatCompletion.create(
            model=params.gpt_chat_model,
            temperature=0.7,
            top_p=0.9,
            max_tokens=params.max_tokens,
            messages=messages_history
        )
        return response.choices[0].message.content.strip()
    except openai.error.RateLimitError:
        return "OpenAI API rate limit exceeded! Please try again later."
    except Exception as e:
        return f"OpenAI API error: {e}"


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
