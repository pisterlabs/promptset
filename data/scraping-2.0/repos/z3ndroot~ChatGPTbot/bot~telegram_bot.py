import asyncio
import logging
import os

from aiogram import Bot
from aiogram import types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ContentType, BotCommand
from aiogram.utils import executor
from aiogram.utils.exceptions import RetryAfter, CantParseEntities, TelegramAPIError

from chatai import GPT
from voicing import Announcer
from openai.error import RateLimitError


class TelegramBot:

    def __init__(self, config: dict, gpt, announcer):
        """
        Bot initialization with the given configuration, gpt object and announcer object
        :param config: dictionary with bot configurations
        :param gpt: GPT object
        :param announcer: Announcer object
        """
        self.storage = MemoryStorage()
        self.bot = Bot(token=config["token_bot"])
        self.allowed_user_ids = config['allowed_user_ids']
        self.bot_command = [
            BotCommand('clear', 'Cleaning up the conversation '),
            BotCommand('system_role',
                       'Provides initial instructions for the model(e.g. /system_role You are a helpful assistant.)'),
            BotCommand('image', 'Generates an image by prompt (e. g. /image car)'),
            BotCommand('help', "I'll show you how to use this bot"),
        ]
        self.dp = Dispatcher(self.bot, storage=self.storage)
        self.in_cor = InlineKeyboardMarkup(row_width=4)
        self.button_clear = InlineKeyboardButton(text="voice", callback_data="voice")
        self.in_cor.add(self.button_clear)
        self.gpt: GPT = gpt
        self.announcer: Announcer = announcer
        self.config = config

    async def _on_startup(self, dp: Dispatcher):
        """
        Run when the bot starts, sends a set of commands
        """
        await dp.bot.set_my_commands(self.bot_command)

    async def _help(self, message: types.Message):
        """
        Shows the help menu.
        """
        commands = [f'/{command.command} - {command.description}' for command in self.bot_command]
        help_message = "👋Welcome, my friend, to ChatGPT bot!" + \
                       " I am an intelligent assistant designed to make your life easier," + \
                       " talk to me Use the following commands to interact with me:" + \
                       "\n\n" + \
                       "\n".join(commands) + \
                       "\n\n" + \
                       "🗣Send me a voice message and I'll convert it into a message" + \
                       "\n" + \
                       "🎧Click the voice button so I can voice my message" + \
                       "\n\n" + \
                       "🙋‍♂️Write your question and see what I can do for you today!"

        await self.bot.send_message(message.from_user.id, help_message)

    async def _chat(self, message: types.Message, text: str = None, audio=False):
        """
        Sending a model response by user message
        """
        await self.bot.send_chat_action(message.from_user.id, "typing")
        text = message.text if not audio else text
        try:
            if self.config['stream']:
                counter = 0
                is_new_message_sent = 0
                content = await message.reply("...")
                stream_response = self.gpt.create_chat_stream(text, chat_id=str(message.from_user.id))
                async for response, tag in stream_response:
                    chunks = self.__text_into_chunks(response)  # splits the text into chunks
                    if len(chunks) == 0:
                        continue
                    chunk = chunks[-1]
                    if len(chunks) > 1 and is_new_message_sent == 0:
                        try:
                            await content.edit_text(chunks[0], reply_markup=self.in_cor,
                                                    parse_mode=types.ParseMode.MARKDOWN)
                        except CantParseEntities:
                            await content.edit_text(chunks[0], reply_markup=self.in_cor)
                        content = await message.reply("...")
                        is_new_message_sent += 1
                        counter = 0
                    try:
                        if not tag and counter % 60 == 0:
                            await content.edit_text(chunk)
                        elif tag:
                            await content.edit_text(chunk, reply_markup=self.in_cor,
                                                    parse_mode=types.ParseMode.MARKDOWN)

                    except RetryAfter as e:
                        logging.warning(e)
                        await asyncio.sleep(e.timeout)
                    except CantParseEntities:
                        await content.edit_text(chunk, reply_markup=self.in_cor)
                    counter += 1
                    await asyncio.sleep(0.01)
            else:

                answer = await self.gpt.create_chat(text, message.from_user.id)
                chunks = self.__text_into_chunks(answer)

                for chunk in chunks:
                    try:
                        await message.reply(chunk, reply_markup=self.in_cor, parse_mode=types.ParseMode.MARKDOWN)
                    except CantParseEntities as e:
                        await message.reply(chunk, reply_markup=self.in_cor)
        except RateLimitError as e:
            logging.error(f'Errors when sending opeanai request: {e}')
            await self.bot.send_message(message.from_user.id, f"Error when requesting: {e}")

    async def error_handler(self, update: types.Update, exception):
        """
        Error handler in the aiogram library
        """
        logging.error(f'Caused by the update error: {exception}')
        return True

    async def _gen_image(self, message: types.Message):
        """
        Sending an image from the model at the user's request
        """
        logging.info(f"New prompt generate image from @{message.from_user.username} (id: {message.from_user.id})")
        prompt = message.text.replace("/image", "")

        if prompt == '':
            await message.reply("You must provide a prompt")
            return

        await self.bot.send_chat_action(message.from_user.id, "upload_photo")
        url_image = await self.gpt.generate_image(prompt)
        if not url_image.startswith("https://"):
            await message.reply(url_image)
            return
        await self.bot.send_photo(message.from_user.id, url_image)

    async def _allowed_users_filter(self, message: types.Message):
        if self.allowed_user_ids == '*':
            self.gpt.create_user_history(f'{message.from_user.id}', f'@{message.from_user.username}')
            return True

        if str(message.from_user.id) in self.allowed_user_ids.split(','):
            self.gpt.create_user_history(f'{message.from_user.id}', f'@{message.from_user.username}')
            return True

        return False

    async def _voicing(self, callback: types.CallbackQuery):
        """
        Processing the inline button and sending the generated voice
        """
        logging.info(
            f'Request to be converted into audio from {callback.from_user.username} (id: {callback.from_user.id})')
        await self.bot.send_chat_action(callback.from_user.id, 'record_voice')
        voices = await self.announcer.voicing(callback.message.text, callback.from_user.id)
        for voice in voices:
            if voice is None:
                await self.bot.send_message(callback.from_user.id, "Unfortunately, I can't recognize this message")
                return
            await self.bot.send_chat_action(callback.from_user.id, 'upload_voice')
            ogg_file = types.InputFile(voice)
            try:
                await self.bot.send_voice(callback.from_user.id, ogg_file)
            except TelegramAPIError as e:
                logging.warning(e)
            finally:
                os.remove(voice)

    async def _clear_chat(self, message: types.Message):
        """
        Clear command processing
        """
        logging.info(f"Clear history from @{message.from_user.username} (id: {message.from_user.id})")
        self.gpt.clear_history(chat_id=str(message.from_user.id))
        await self.bot.send_message(message.from_user.id, "History brushed off✅")

    async def _get_system_message_for_user(self, message: types.Message):
        """
        Processing the system_role command
        """
        text = message.text.replace("/system_message", "")
        self.gpt.system_message(text, chat_id=str(message.from_user.id))
        await self.bot.send_message(message.from_user.id, "Complete✅")

    async def _audio_to_chat(self, audio: types.Message):
        """
        Voice message processing
        """
        logging.info(f"New audio received from user @{audio.from_user.username} (id: {audio.from_user.id})")
        await audio.voice.download(destination_file=f'audio/{audio.from_user.id}.ogg')
        await self.gpt.convert_audio(chat_id=str(audio.from_user.id))
        text = await self.gpt.transcriptions(chat_id=str(audio.from_user.id))
        await self._chat(audio, text, audio=True)
        await self.gpt.delete_audio(chat_id=audio.from_user.id)

    async def _message(self, message: types.Message):
        """
        Processing text messages
        """
        logging.info(f"New message received from user @{message.from_user.username} (id: {message.from_user.id})")
        await self._chat(message)

    def __text_into_chunks(self, text: str, split_size: int = 4096) -> list[str]:
        return [text[i:i + split_size] for i in range(0, len(text), split_size)]

    def _reg_handler(self, dp: Dispatcher):
        """
        registration of message handlers
        """
        dp.register_message_handler(self._help, self._allowed_users_filter, commands="start")
        dp.register_message_handler(self._help, self._allowed_users_filter, commands="help")
        dp.register_message_handler(self._clear_chat, self._allowed_users_filter, commands="clear")
        dp.register_callback_query_handler(self._voicing, self._allowed_users_filter, text="voice")
        dp.register_message_handler(self._get_system_message_for_user, self._allowed_users_filter,
                                    commands="system_role")
        dp.register_message_handler(self._gen_image, self._allowed_users_filter, commands="image")
        dp.register_message_handler(self._audio_to_chat, self._allowed_users_filter, content_types=ContentType.VOICE)
        dp.register_message_handler(self._message, self._allowed_users_filter)
        dp.register_errors_handler(self.error_handler)

    def run(self):
        """
        bot startup
        """
        self._reg_handler(self.dp)
        executor.start_polling(self.dp, skip_updates=True, on_startup=self._on_startup)
