import os
import asyncio
import config
from pyrogram import filters,enums
import openai
from datetime import datetime
from data.database import db
from data.openAi import generate_images,is_content_acceptable,transcribe_audio,ChatGPT
from bot import app,user_tasks,user_semaphores,command_list
from  .functions import add_startup_user,is_answered

async def generate_image_handle(message):
    user_id = message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    prompt = message.text
    await app.send_chat_action(chat_id=message.chat.id,action=enums.ChatAction.UPLOAD_PHOTO)
    try:
        image_url = await generate_images(prompt, n_images=1)
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await message.reply_text(text,disable_web_page_preview=True)
            return
        else:
            raise
    await app.send_chat_action(chat_id=message.chat.id,action=enums.ChatAction.UPLOAD_PHOTO)
    await message.reply_photo(image_url)


async def gen_text(message,prompt,use_new_dialog_timeout=True):
        user_id = message.from_user.id
        if prompt in command_list:return
        if prompt.startswith("/img"):return
        if await is_answered(message):return
        balance = db.get_user_attribute(user_id, "user_balance")
        chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
        if chat_mode == "artist":
            await generate_image_handle(message)
            curent_credits = balance - 50
            db.set_user_attribute(user_id, "user_balance", curent_credits)
            return
        async def message_handle_fn():
            if use_new_dialog_timeout:
                if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                    db.start_new_dialog(user_id)
                    await message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=enums.ParseMode.HTML)
            db.set_user_attribute(user_id, "last_interaction", datetime.now())
            chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
            balance = db.get_user_attribute(user_id, "user_balance")
            # in case of CancelledError
            n_input_tokens, n_output_tokens = 0, 0
            current_model = db.get_user_attribute(user_id, "current_model")

            try:
                await app.send_chat_action(chat_id=message.chat.id,action=enums.ChatAction.TYPING)
                placeholder_message = await message.reply_text("...")
                if balance <= 0:return await placeholder_message.edit("You don't have enough credit. Buy more /payment.")
                dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
                parse_mode = {
                    "html": enums.ParseMode.HTML,
                    "markdown": enums.ParseMode.MARKDOWN
                }[config.chat_modes[chat_mode]["parse_mode"]]

                chatgpt_instance = ChatGPT(model=current_model)
                if config.enable_message_streaming:
                    gen = chatgpt_instance.send_message_stream(prompt,
                     chat_mode,
                     dialog_messages=dialog_messages, )
                else:
                    answer,(n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(prompt,
                                          chat_mode=chat_mode,         dialog_messages=dialog_messages)
                    print(n_input_tokens, n_output_tokens)
                    async def fake_gen():
                        yield "finished", answer,(n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                    gen = fake_gen()
                prev_answer = ""
                async for gen_item in gen:
                    status, answer,(n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item
                    print(n_input_tokens, n_output_tokens)
                    answer = answer[:4096]
                    if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                        continue
                    try:
                        await app.edit_message_text(chat_id =message.chat.id, message_id = placeholder_message.id,text = answer, parse_mode=parse_mode)
                    except Exception as e:
                        if str(e).startswith("Message is not modified"):
                            continue
                        else:
                            await app.edit_message_text(chat_id =message.chat.id, message_id = placeholder_message.id,text = answer)
                    await asyncio.sleep(0.01)
                    prev_answer = answer
                new_dialog_message = {"user": prompt, "bot": answer, "date": datetime.now()}
                curent_credits = balance - n_output_tokens
                db.set_user_attribute(user_id, "user_balance", curent_credits)
                db.set_dialog_messages(
                    user_id,
                    db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                    dialog_id=None
            )
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                error_text = f"Something went wrong during completion. Reason: {e}"
                await message.reply_text(error_text)
                return
            

            if n_first_dialog_messages_removed > 0:
                if n_first_dialog_messages_removed == 1:
                    text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
                else:
                    text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
                await message.reply_text(text)

        async with user_semaphores[user_id]:
            task = asyncio.create_task(message_handle_fn())
            user_tasks[user_id] = task
            try:
                await task
            except asyncio.CancelledError:
                await message.reply_text("✅ Canceled")
            else:
                pass
            finally:
                if user_id in user_tasks:
                    del user_tasks[user_id]