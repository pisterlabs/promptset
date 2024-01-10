import asyncio
from datetime import datetime
import html
import io
import json
import logging
import traceback
import openai
from telegram import Update, error as telegram_error
from telegram.constants import ParseMode
from telegram.ext import CallbackContext
from functools import wraps

import openai_utils
from database import Database

from user_utils import UserUtils
import bot
import config

_user = UserUtils(Database())
_logger = logging.getLogger(__name__)

HELP_MESSAGE = """Commands:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /mode – Select chat mode
⚪ /settings – Show settings
⚪ /balance – Show balance
⚪ /help – Show help

🎨 Generate images from text prompts in <b>👩‍🎨 Artist</b> /mode
👥 Add bot to <b>group chat</b>: /help_group_chat
🎤 You can send <b>Voice Messages</b> instead of text
"""

HELP_GROUP_CHAT_MESSAGE = """You can add bot to any <b>group chat</b> to help and entertain its participants!

Instructions (see <b>video</b> below):
1. Add the bot to the group chat
2. Make it an <b>admin</b>, so that it can see messages (all other rights can be restricted)
3. You're awesome!

To get a reply from the bot in the chat – @ <b>tag</b> it or <b>reply</b> to its message.
For example: "{bot_username} write a poem about Telegram"
"""


def ensure_user_registered(func):
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        await _user.register_user_if_not_exists(update, context, update.effective_user)
        return await func(update, context, *args, **kwargs)

    return wrapper


def update_user_interaction(func):
    async def wrapper(update: Update, context: CallbackContext, *args, **kwargs):
        user_id = update.effective_user.id
        await _user.update_user_last_interaction(user_id)
        return await func(update, context, *args, **kwargs)

    return wrapper


async def _is_bot_mentioned(update: Update, context: CallbackContext):
    try:
        message = update.message

        if message.chat.type == "private":
            return True

        if message.text is not None and ("@" + context.bot.username) in message.text:
            return True

        if message.reply_to_message is not None:
            if message.reply_to_message.from_user.id == context.bot.id:
                return True
    except:
        return True
    else:
        return False


@ensure_user_registered
async def _is_previous_message_not_answered_yet(
    update: Update, context: CallbackContext
):
    user_id = update.message.from_user.id
    if _user.user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(
            text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML
        )
        return True
    else:
        return False


@ensure_user_registered
@update_user_interaction
async def _message_handle_fn(
    update: Update,
    context: CallbackContext,
    message: str,
    use_new_dialog_timeout: bool = True,
):
    user_id = update.message.from_user.id
    chat_mode = await _user.get_current_chat_mode(user_id)
    # new dialog timeout
    if use_new_dialog_timeout:
        if (
            datetime.now() - await _user.get_last_interaction(user_id)
        ).seconds > config.new_dialog_timeout and len(
            await _user.get_dialog_messages(user_id)
        ) > 0:
            await _user.start_new_dialog(user_id)
            await update.message.reply_text(
                f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅",
                parse_mode=ParseMode.HTML,
            )

    # in case of CancelledError
    n_input_tokens, n_output_tokens = 0, 0
    current_model = await _user.get_current_model(user_id)

    try:
        # send placeholder message to user
        placeholder_message = await update.message.reply_text("...")

        # send typing action
        await update.message.chat.send_action(action="typing")

        if message is None or len(message) == 0:
            await update.message.reply_text(
                "🥲 You sent <b>empty message</b>. Please, try again!",
                parse_mode=ParseMode.HTML,
            )
            return

        dialog_messages = await _user.get_dialog_messages(user_id)
        parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
            config.chat_modes[chat_mode]["parse_mode"]
        ]

        chatgpt_instance = openai_utils.ChatGPT(model=current_model)
        if config.enable_message_streaming:
            gen = chatgpt_instance.send_message_stream(
                message, dialog_messages=dialog_messages, chat_mode=chat_mode
            )
        else:
            (
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = await chatgpt_instance.send_message(
                message, dialog_messages=dialog_messages, chat_mode=chat_mode
            )

            async def fake_gen():
                yield "finished", answer, (
                    n_input_tokens,
                    n_output_tokens,
                ), n_first_dialog_messages_removed

            gen = fake_gen()

        prev_answer = ""
        async for gen_item in gen:
            (
                status,
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = gen_item

            answer = answer[:4096]  # telegram message limit

            # update only when 100 new symbols are ready
            if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                continue

            try:
                await context.bot.edit_message_text(
                    answer,
                    chat_id=placeholder_message.chat_id,
                    message_id=placeholder_message.message_id,
                    parse_mode=parse_mode,
                )
            except telegram_error.BadRequest as e:
                if str(e).startswith("Message is not modified"):
                    continue
                else:
                    await context.bot.edit_message_text(
                        answer,
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                    )

            await asyncio.sleep(0.01)  # wait a bit to avoid flooding

            prev_answer = answer

        # update user data
        new_dialog_message = {
            "user": message,
            "bot": answer,
            "date": datetime.now(),
        }
        await _user.set_dialog_messages(
            user_id, await _user.get_dialog_messages(user_id) + [new_dialog_message]
        )

        await _user.update_n_used_tokens(
            user_id, current_model, n_input_tokens, n_output_tokens
        )
    except asyncio.CancelledError:
        # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
        await _user.update_n_used_tokens(
            user_id, current_model, n_input_tokens, n_output_tokens
        )
        raise
    except Exception as e:
        error_text = f"Something went wrong during completion. Reason: {e}"
        _logger.error(error_text)
        await update.message.reply_text(error_text)
        return

    # send message if some messages were removed from the context
    if n_first_dialog_messages_removed > 0:
        if n_first_dialog_messages_removed == 1:
            text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
        else:
            text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def _generate_image_handle(
    update: Update, context: CallbackContext, message=None
):
    if await _is_previous_message_not_answered_yet(update, context):
        return

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(
            message, n_images=config.return_n_generated_images, size=config.image_size
        )
    except openai.BadRequestError as e:
        if str(e).startswith(
            "Your request was rejected as a result of our safety system"
        ):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    user_id = update.message.from_user.id
    await _user.set_n_generated_images(
        user_id,
        config.return_n_generated_images + await _user.get_n_generated_images(user_id),
    )

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)


async def _vision_message_handle_fn(
    update: Update, context: CallbackContext, use_new_dialog_timeout: bool = True
):
    user_id = update.message.from_user.id
    chat_mode = await _user.get_current_chat_mode(user_id)
    current_model = await _user.get_current_model(user_id)

    if current_model != "gpt-4-vision-preview":
        await update.message.reply_text(
            "🥲 Images processing is only available for <b>gpt-4-vision-preview</b> model. Please change your settings in /settings",
            parse_mode=ParseMode.HTML,
        )
        return

    # new dialog timeout
    if use_new_dialog_timeout:
        if (
            datetime.now() - await _user.get_last_interaction(user_id)
        ).seconds > config.new_dialog_timeout and len(
            await _user.get_dialog_messages(user_id)
        ) > 0:
            await _user.start_new_dialog(user_id)
            await update.message.reply_text(
                f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅",
                parse_mode=ParseMode.HTML,
            )

    photo = update.message.effective_attachment[-1]
    photo_file = await context.bot.get_file(photo.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await photo_file.download_to_memory(buf)
    buf.name = "image.jpg"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    # in case of CancelledError
    n_input_tokens, n_output_tokens = 0, 0

    try:
        # send placeholder message to user
        placeholder_message = await update.message.reply_text("...")
        message = update.message.caption

        # send typing action
        await update.message.chat.send_action(action="typing")

        if message is None or len(message) == 0:
            await update.message.reply_text(
                "🥲 You sent <b>empty message</b>. Please, try again!",
                parse_mode=ParseMode.HTML,
            )
            return

        dialog_messages = await _user.get_dialog_messages(user_id)
        parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
            config.chat_modes[chat_mode]["parse_mode"]
        ]

        chatgpt_instance = openai_utils.ChatGPT(model=current_model)
        if config.enable_message_streaming:
            gen = chatgpt_instance.send_vision_message_stream(
                message,
                dialog_messages=dialog_messages,
                image_buffer=buf,
                chat_mode=chat_mode,
            )
        else:
            (
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = await chatgpt_instance.send_vision_message(
                message,
                dialog_messages=dialog_messages,
                image_buffer=buf,
                chat_mode=chat_mode,
            )

            async def fake_gen():
                yield "finished", answer, (
                    n_input_tokens,
                    n_output_tokens,
                ), n_first_dialog_messages_removed

            gen = fake_gen()

        prev_answer = ""
        async for gen_item in gen:
            (
                status,
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = gen_item

            answer = answer[:4096]  # telegram message limit

            # update only when 100 new symbols are ready
            if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                continue

            try:
                await context.bot.edit_message_text(
                    answer,
                    chat_id=placeholder_message.chat_id,
                    message_id=placeholder_message.message_id,
                    parse_mode=parse_mode,
                )
            except telegram_error.BadRequest as e:
                if str(e).startswith("Message is not modified"):
                    continue
                else:
                    await context.bot.edit_message_text(
                        answer,
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                    )

            await asyncio.sleep(0.01)  # wait a bit to avoid flooding

            prev_answer = answer

        # update user data
        new_dialog_message = {
            "user": message,
            "bot": answer,
            "date": datetime.now(),
        }
        await _user.set_dialog_messages(
            user_id, await _user.get_dialog_messages(user_id) + [new_dialog_message]
        )

        await _user.update_n_used_tokens(
            user_id, current_model, n_input_tokens, n_output_tokens
        )
    except asyncio.CancelledError:
        # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
        await _user.update_n_used_tokens(
            user_id, current_model, n_input_tokens, n_output_tokens
        )
        raise
    except Exception as e:
        error_text = f"Something went wrong during completion. Reason: {e}"
        _logger.error(error_text)
        await update.message.reply_text(error_text)
        return


@ensure_user_registered
@update_user_interaction
async def help_group_chat_handle(update: Update, context: CallbackContext):
    text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    await update.message.reply_video(config.help_group_chat_video_path)


@ensure_user_registered
@update_user_interaction
async def start_handle(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    await _user.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)


@ensure_user_registered
@update_user_interaction
async def help_handle(update: Update, context: CallbackContext):
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


@ensure_user_registered
@update_user_interaction
async def retry_handle(update: Update, context: CallbackContext):
    if await _is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    dialog_messages = await _user.get_dialog_messages(user_id)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    await _user.set_dialog_messages(
        user_id, dialog_messages
    )  # last message was removed from the context

    await message_handle(
        update,
        context,
        message=last_dialog_message["user"],
        use_new_dialog_timeout=False,
    )


@ensure_user_registered
@update_user_interaction
async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await _is_bot_mentioned(update, context):
        return

    if await _is_previous_message_not_answered_yet(update, context):
        return

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    user_id = update.message.from_user.id
    await _user.update_n_transcribed_seconds(user_id, voice.duration)

    await message_handle(update, context, message=transcribed_text)


@ensure_user_registered
@update_user_interaction
async def new_dialog_handle(update: Update, context: CallbackContext):
    if await _is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    await _user.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = await _user.get_current_chat_mode(user_id)
    await update.message.reply_text(
        f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML
    )


@ensure_user_registered
@update_user_interaction
async def cancel_handle(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    if user_id in _user.user_tasks:
        task = _user.user_tasks[user_id]
        task.cancel()
        del _user.user_tasks[user_id]  # Remove the canceled task from the dictionary
        await update.message.reply_text(
            "<i>Task canceled successfully.</i>", parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            "<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML
        )


@ensure_user_registered
@update_user_interaction
async def show_chat_modes_handle(update: Update, context: CallbackContext):
    if await _is_previous_message_not_answered_yet(update, context):
        return

    text, reply_markup = bot.get_chat_mode_menu(0)
    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


@ensure_user_registered
@update_user_interaction
async def message_handle(
    update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True
):
    # check if bot was mentioned (for group chats)
    if not await _is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text or update.message.caption

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    if await _is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    chat_mode = await _user.get_current_chat_mode(user_id)

    if chat_mode == "artist":
        await _generate_image_handle(update, context, message=_message)
        return

    if update.message.photo is not None and len(update.message.photo) > 0:
        await _vision_message_handle_fn(update, context, use_new_dialog_timeout)
        return

    async with _user.user_semaphores[user_id]:
        task = asyncio.create_task(
            _message_handle_fn(update, context, _message, use_new_dialog_timeout)
        )
        _user.user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in _user.user_tasks:
                del _user.user_tasks[user_id]


@ensure_user_registered
@update_user_interaction
async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
    if await _is_previous_message_not_answered_yet(update.callback_query, context):
        return

    query = update.callback_query
    await query.answer()

    page_index = int(query.data.split("|")[1])
    if page_index < 0:
        return

    text, reply_markup = bot.get_chat_mode_menu(page_index)
    try:
        await query.edit_message_text(
            text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )
    except telegram_error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


@ensure_user_registered
@update_user_interaction
async def set_chat_mode_handle(update: Update, context: CallbackContext):
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    await _user.set_current_chat_mode(user_id, chat_mode)
    await _user.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML,
    )


@ensure_user_registered
@update_user_interaction
async def settings_handle(update: Update, context: CallbackContext):
    if await _is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id

    text, reply_markup = await bot.get_settings_menu(user_id)
    await update.message.reply_text(
        text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
    )


@ensure_user_registered
@update_user_interaction
async def set_settings_handle(update: Update, context: CallbackContext):
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    await _user.set_current_model(user_id, model_key)
    await _user.start_new_dialog(user_id)

    text, reply_markup = await bot.get_settings_menu(user_id)
    try:
        await query.edit_message_text(
            text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )
    except telegram_error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


@ensure_user_registered
@update_user_interaction
async def show_balance_handle(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = await _user.get_n_used_tokens(user_id)
    n_generated_images = await _user.get_n_generated_images(user_id)
    n_transcribed_seconds = await _user.get_n_transcribed_seconds(user_id)

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = (
            n_used_tokens_dict[model_key]["n_input_tokens"],
            n_used_tokens_dict[model_key]["n_output_tokens"],
        )
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key][
            "price_per_1000_input_tokens"
        ] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key][
            "price_per_1000_output_tokens"
        ] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = (
        config.models["info"]["dalle-2"]["price_per_1_image"] * n_generated_images
    )
    if n_generated_images != 0:
        details_text += f"- DALL·E 2 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"][
        "price_per_1_min"
    ] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars

    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


@ensure_user_registered
async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


@ensure_user_registered
async def error_handle(update: Update, context: CallbackContext) -> None:
    _logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(
            None, context.error, context.error.__traceback__
        )
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        def split_text_into_chunks_fn(text, chunk_size):
            for i in range(0, len(text), chunk_size):
                yield text[i : i + chunk_size]

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks_fn(message, 4096):
            try:
                await context.bot.send_message(
                    update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML
                )
            except telegram_error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(
            update.effective_chat.id, "Some error in error handler"
        )
