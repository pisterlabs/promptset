import os
import asyncio
import config
from pyrogram import (
    filters, 
    enums)

from datetime import datetime
from pyrogram.types import (
    InlineKeyboardMarkup, 
    InlineKeyboardButton, 
    ReplyKeyboardMarkup, 
    KeyboardButton,
    CallbackQuery)

from bot import (
    app,
    user_tasks,
    user_semaphores,
    command_list
)
from  .functions import(
    add_startup_user,
    is_answered
)
from  .response import(
    gen_text,
    generate_image_handle
)
import openai
from data.database import db
from data.openAi import (
    generate_images,
    is_content_acceptable,
    transcribe_audio,
    ChatGPT
)

button = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Share the link ↗️", url=f"https://t.me/share/url?url=https://t.me/OpenAiXChatBot"),
                InlineKeyboardButton("📝 Subscription", callback_data="crypto"),
            ]
        ]
    )

async def add_user_(message):
    user = message.from_user
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name,
            )
        db.start_new_dialog(user.id)
    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

@app.on_message(filters.command("payment"))
async def balance_handler(_, message):
    await add_user_(message)
    await message.reply_text("make payment",
                            reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("💳Visa/Mastercard/etc", callback_data=f"card")],
                        [InlineKeyboardButton("💎Crypto", callback_data=f"crypto")]
                    ]))
    
@app.on_message(filters.command("start"))
async def start_message_handler(_, message):
    await add_user_(message)
    start_message = "Hi! I am **chatGPT** Telegram bot\n\n"
    start_message += config.help_message
    await message.reply_text(start_message,disable_web_page_preview=True)

@app.on_message(filters.command("help"))
async def help_message_handler(_, message):
    await add_user_(message)
    await message.reply_text(config.help_message,disable_web_page_preview=True)

@app.on_message(filters.command("retry"))
async def retry_message_handler(_, message):
    await add_user_(message)
    if await is_answered(message):return
    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    print(dialog_messages)
    if len(dialog_messages) == 0:
        await message.reply_text("<i>No message to retry 🤷‍♂️</i>")
        return
    prompt = dialog_messages[0]['user']
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)
    await gen_text(message,prompt,use_new_dialog_timeout=False)
    
@app.on_message(filters.command("cancel"))
async def cancel_message_handler(_, message):
    await add_user_(message)
    user_id = message.from_user.id
    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await message.reply_text("<i>Nothing to cancel...</i>")

@app.on_message(filters.command("new"))
async def new_message_handler(_, message):
    await add_user_(message)
    if await is_answered(message): return
    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)
    await message.reply_text("<i>Starting new dialog ✅</i>")
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}")


def get_chat_mode_menu(page_index: int, user_id: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]
    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
        keyboard.append(
            [
                InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}"),
            ]
        )
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))
        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("⏩", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("⏪", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("⏪", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("⏩", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
    reply_markup = InlineKeyboardMarkup(keyboard)
    return text, reply_markup

@app.on_callback_query(filters.regex("set_chat_mode"))
async def set_chat_mode_handler(_, cq: CallbackQuery):
    user_id = cq.from_user.id
    print(user_id)
    chat_mode = cq.data.split("|")[1]
    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    text, reply_markup = get_chat_mode_menu(0,user_id)
    try:
        await cq.edit_message_text(text, reply_markup=reply_markup)
    except:
        pass
    await app.send_message(
        cq.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}")
    
@app.on_callback_query(filters.regex("show_chat_modes"))
async def show_chat_modes_handler(_, cq: CallbackQuery):
    await add_user_(cq.message)
    if await is_answered(cq.message):return
    user_id = cq.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    page_index = int(cq.data.split("|")[1])
    if page_index < 0:
         return
    text, reply_markup = get_chat_mode_menu(page_index,user_id)
    try:
        await cq.edit_message_text(text, reply_markup=reply_markup)
    except:
        pass

@app.on_message(filters.command("mode"))
async def mode_message_handler(_, message):
    await add_user_(message)
    if await is_answered(message):return
    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    text, reply_markup = get_chat_mode_menu(0,user_id)
    await message.reply_text(text, reply_markup=reply_markup)


@app.on_message(filters.command("img"))
async def img_message_handler(_, message):
    await add_user_(message)
    if await is_answered(message):return
    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)
    await app.send_chat_action(chat_id=message.chat.id,action=enums.ChatAction.UPLOAD_PHOTO)
    prompt = message.text.split(None, 1)[1].strip()
    try:
        image_url = await generate_images(prompt, n_images=1)
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await message.reply_text(text)
            return
        else:
            raise
    await message.reply_photo(image_url)

@app.on_message(filters.incoming,group=1)
async def incoming_message_handler(_,message):
    await add_user_(message)
    prompt = message.text
    if await is_answered(message):return
    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    user_id = message.from_user.id
    await gen_text(message,prompt,use_new_dialog_timeout=False)
    
@app.on_edited_message(filters.private,group=2)
async def edit_message_handler(_, message):
    print(message)
    await add_user_(message)
    text = "📝 Unfortunately, message <b>editing</b> is not supported"
    await message.reply_text(text,)
    
@app.on_message(filters.command("balance"))
async def balance_handler(_, message):
    await add_user_(message)
    balance = db.get_user_attribute(message.from_user.id, "user_balance")
    await message.reply_text(f"{balance} tokens",
                            reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("💳Visa/Mastercard/etc", callback_data=f"card")],
                        [InlineKeyboardButton("💎Crypto", callback_data=f"crypto")]
                    ]))