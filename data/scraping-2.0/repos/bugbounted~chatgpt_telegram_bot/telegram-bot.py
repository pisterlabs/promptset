import logging
import re
import threading
import time
from typing import Optional, Tuple

import openai
from telegram import ChatMember, ChatMemberUpdated, Update, Bot
from telegram import ParseMode
from telegram.ext import Updater, ChatMemberHandler, MessageHandler, Filters, ContextTypes, CommandHandler

from handlers.commands import help_command, cancel_command, save_forwarded_message, clear_forwarded_message
from settings import debug, main_user_id, available_in_chat, tgkey, botname, minutes_for_user_thinking


class GreetedUser:
    def __init__(self, user_id: int, greeting_bot_message: str, greeting_date: float):
        self.user_id = user_id
        self.greeting_bot_message = greeting_bot_message
        self.greeting_date = greeting_date


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

completion = openai.Completion()

last_greeted_user = dict()


##################
# Command handlers#
##################
def remove_all_mentions(text):
    return re.sub(r'@\w+', '', text)


def send_generated_image(update):
    question = update.message.text.replace("/generateimage", "").replace(f'@{botname}', "")
    if debug:
        print("Input:\n" + question)
    answer = generate_image(question)
    update.message.reply_photo(answer)


def answer_a_question(update):
    text_to_reply = update.message.text.replace(f'@{botname}', "")
    answer = generate_answer(update.message.from_user.id, text_to_reply)
    update.effective_chat.send_message(f'@{update.message.from_user.username} {answer}')


def reply_a_question(update):
    text_to_reply = update.message.reply_to_message.text + " " + update.message.text.replace(f'@{botname}', "")
    answer = generate_answer(update.message.reply_to_message.from_user.id, text_to_reply)
    update.message.reply_text(text=f'@{update.message.from_user.username} {answer}',
                              reply_to_message_id=update.message.reply_to_message.message_id)


def simple_reply(update, bot_message=None):
    if bot_message is None:
        bot_message = update.message.reply_to_message.text
    prompt = f'AI:{bot_message}\nHuman:{update.message.text}\nAI:'
    answer = generate_answer_raw(update.message.from_user.id, prompt)
    update.message.reply_text(text=answer)


def answer_user_in_chat(context: ContextTypes, chat: str):
    if chat not in available_in_chat:
        return
    bot = Bot(token=tgkey)
    answer = generate_answer(context.user_data["reply_user_id"], context.user_data["text"])
    if debug:
        print("answer_user_in_chat:\n" + f'chat_id=@{chat}\n @{context.user_data["mention_markdown"]} {answer}')

    bot.send_message(chat_id=f'@{chat}',
                     text=f'@{context.user_data["mention_markdown"]} {answer}',
                     parse_mode=ParseMode.MARKDOWN)
    clear_forwarded_message(context)


# Detect if user answered to greeting bot message without mentioning bot or reply to bot message
def is_need_answer(update: Update) -> bool:
    if update.message.chat.type == "private" or \
            update.message.chat.username not in available_in_chat or \
            "@" in update.message.text or \
            update.message.reply_to_message is not None or \
            last_greeted_user.get(update.message.chat.id) is None or \
            last_greeted_user[update.message.chat.id].user_id != update.message.from_user.id or \
            (time.time() - last_greeted_user[update.message.chat.id].greeting_date) / 60 > minutes_for_user_thinking:
        return False
    else:
        return True


def message_handler(update: Update, context: ContextTypes):
    if update.message.chat.type == "private":
        if update.message.from_user.id != main_user_id:
            update.effective_chat.send_message("❌ Для этого действия нужно быть администратором бота")
            return
        elif update.message.forward_date is not None:
            if save_forwarded_message(update, context):
                update.effective_chat.send_message("Напишите название чата в котором нужно ответить пользователю")
            else:
                update.effective_chat.send_message("Пользователь скрыл данные, невозможно ответить на его сообщение")
            return
        elif context.user_data.get("text") is not None and context.user_data["text"] != "":
            comput = threading.Thread(target=answer_user_in_chat, args=(context, update.message.text.replace("@", ""),))
            comput.start()
            return
    elif update.message.chat.username not in available_in_chat:
        return

    if f'@{botname}' in update.message.text:
        if update.message.reply_to_message is None:
            if "/generateimage" in update.message.text:
                comput = threading.Thread(target=send_generated_image, args=(update,))
                comput.start()
            else:
                comput = threading.Thread(target=answer_a_question, args=(update,))
                comput.start()
        else:
            comput = threading.Thread(target=reply_a_question, args=(update,))
            comput.start()
    elif update.message.reply_to_message is not None and update.message.reply_to_message.from_user.username == botname:
        """Reply to a message."""
        comput = threading.Thread(target=simple_reply, args=(update,))
        comput.start()
    else:
        if update.message.from_user.id == main_user_id and update.message.chat.type == "private":
            comput = threading.Thread(target=answer_a_question, args=(update,))
            comput.start()
        elif is_need_answer(update):
            comput = threading.Thread(target=simple_reply,
                                      args=(update, last_greeted_user[update.message.chat.id].greeting_bot_message))
            comput.start()
            last_greeted_user.pop(update.message.chat.id)


def extract_status_change(chat_member_update: ChatMemberUpdated) -> Optional[Tuple[bool, bool]]:
    """Takes a ChatMemberUpdated instance and extracts whether the 'old_chat_member' was a member
    of the chat and whether the 'new_chat_member' is a member of the chat. Returns None, if
    the status didn't change.
    """
    status_change = chat_member_update.difference().get("status")
    old_is_member, new_is_member = chat_member_update.difference().get("is_member", (None, None))

    if status_change is None:
        return None

    old_status, new_status = status_change
    was_member = old_status in [
        ChatMember.MEMBER,
        ChatMember.ADMINISTRATOR,
    ] or (old_status == ChatMember.RESTRICTED and old_is_member is True)
    is_member = new_status in [
        ChatMember.MEMBER,
        ChatMember.ADMINISTRATOR,
    ] or (new_status == ChatMember.RESTRICTED and new_is_member is True)

    return was_member, is_member


def greet_chat_members_handler(update, context):
    if debug:
        print("greet_chat_members")

    if update.chat_member.chat.username not in available_in_chat:
        return

    """Greets new users in chats and announces when someone leaves"""
    result = extract_status_change(update.chat_member)
    if result is None:
        return

    was_member, is_member = result

    if not was_member and is_member:
        compute = threading.Thread(target=send_greet_chat_message, args=(update,))
        compute.start()


def send_greet_chat_message(update):
    answer = generate_answer(update.chat_member.new_chat_member.user.id,
                             f'Весело попроси пользователя {update.chat_member.new_chat_member.user.first_name} рассказать о себе и своём опыте в программировании.')
    last_greeted_user[update.chat_member.chat.id] = GreetedUser(update.chat_member.new_chat_member.user.id,
                                                                answer,
                                                                time.time())
    update.effective_chat.send_message(
        f'@{update.chat_member.new_chat_member.user.mention_markdown()} {answer}',
        parse_mode=ParseMode.MARKDOWN
    )


################
# Main functions#
################
def generate_answer(user_id, question):
    prompt = f'Human: {question}\nAI:'
    return generate_answer_raw(user_id, prompt)


def generate_answer_raw(user_id, prompt):
    user_id_str = str(user_id)
    if debug:
        print("----Start generating------")
        print("User: " + user_id_str, "\nQuestion: " + prompt)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"],
        user=user_id_str
    )
    answer = response.choices[0].text.strip()
    return answer


def generate_image(question):
    response = openai.Image.create(
        prompt=question,
        n=1,
        size="256x256"
    )
    return response['data'][0]['url']


#####################
# End main functions#
#####################


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update)


def main():
    """Start the bot."""
    updater = Updater(tgkey)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("cancel", cancel_command))
    dp.add_handler(MessageHandler(Filters.text, message_handler))
    dp.add_handler(ChatMemberHandler(greet_chat_members_handler, ChatMemberHandler.CHAT_MEMBER))
    # log all errors
    dp.add_error_handler(error)
    # Start the Bot
    updater.start_polling(allowed_updates=Update.ALL_TYPES)
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
