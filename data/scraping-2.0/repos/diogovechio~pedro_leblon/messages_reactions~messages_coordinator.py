import asyncio
import random

from constants.constants import MOCK_EDITS
from data_classes.react_data import ReactData
from data_classes.received_message import MessageReceived, TelegramMessage
from messages_reactions.mock_users import mock_users
from pedro_leblon import FakePedro
from messages_reactions.ai_reactions import openai_reactions
from messages_reactions.bot_commands import bot_commands
from messages_reactions.general_text_reactions import words_reactions
from messages_reactions.image_reactions import image_reactions
from utils.logging_utils import telegram_logging, elapsed_time, async_elapsed_time
from utils.openai_utils import extract_website_paragraph_content
from utils.text_utils import https_url_extract, create_username


async def messages_coordinator(
        bot: FakePedro,
        incoming: MessageReceived
) -> None:
    if incoming.message is not None:
        message = incoming.message

        from_debug_chats = message.chat.id in (-20341310, 8375482, -4098496372)

        react_data = await _pre_processor(
            bot=bot,
            message=message,
            from_samuel=message.from_.is_premium
        )

        if message.chat.id in bot.allowed_list:
            if str(message.from_.id) not in bot.config.ignore_users and message.from_.username not in bot.config.ignore_users:
                if message.photo and message.chat.id not in bot.config.not_internal_chats:
                    bot.loop.create_task(
                        image_reactions(
                            bot=bot,
                            message=message,
                            method='cropper' if react_data.from_samuel or from_debug_chats else 'face_classifier',
                            always_send_crop=from_debug_chats
                        )
                    )

                if message.text or message.caption:
                    message.text = message.caption if message.caption else message.text

                    await asyncio.gather(
                        openai_reactions(data=react_data),
                        words_reactions(data=react_data),
                        bot_commands(data=react_data),
                        mock_users(data=react_data),
                    )

        elif not bot.debug_mode:
            bot.loop.create_task(
                bot.leave_chat(
                    chat_id=message.chat.id
                )
            )

            bot.loop.create_task(
                bot.send_message(
                    chat_id=-704277411,
                    message_text=f"new chat id: {incoming.message.chat.id}"
                )
            )

    elif (
            incoming.edited_message is not None
            and incoming.edited_message.chat.id not in bot.config.not_internal_chats
            and incoming.edited_message.edit_date - incoming.edited_message.date < 120
            and random.random() < bot.config.random_params.random_mock_frequency
    ):
        bot.loop.create_task(
            bot.send_message(
                message_text=random.choice(MOCK_EDITS),
                chat_id=incoming.edited_message.chat.id,
                reply_to=incoming.edited_message.message_id
            )
        )

        bot.loop.create_task(telegram_logging(str(incoming)))


@async_elapsed_time
async def _pre_processor(
        bot: FakePedro,
        from_samuel: bool,
        message: TelegramMessage
) -> ReactData:
    url_detector = ""
    input_text = message.text or message.caption

    username = create_username(first_name=message.from_.first_name, username=message.from_.username)
    destroy_message = message.chat.id in bot.config.mock_chats or (
            str(message.from_.id) in bot.config.annoy_users
            or message.from_.username in bot.config.annoy_users
    )

    if message.reply_to_message and message.reply_to_message.text:
        input_text += f" ... o {message.reply_to_message.from_.first_name} tinha dito: " + message.reply_to_message.text

    if input_text is not None:
        if url_detector := await https_url_extract(input_text):
            url_content = await extract_website_paragraph_content(
                url=url_detector,
                session=bot.session
            )

            input_text = input_text.replace(url_detector, url_content)

    return ReactData(
        bot=bot,
        message=message,
        from_samuel=from_samuel,
        username=username,
        input_text=input_text,
        url_detector=url_detector,
        destroy_message=destroy_message,
        mock_chat=message.chat.id in bot.config.mock_chats,
        limited_prompt=(
                str(message.from_.id) in bot.config.limited_prompt_users
                or message.from_.username in bot.config.limited_prompt_users
        )
    )
