import logging
import re
import threading
import time
from typing import Optional, Tuple

import openai
from httpx import ReadTimeout
from openai import BadRequestError
from openai import OpenAI
from openai import RateLimitError
from telegram import ChatMember, ChatMemberUpdated, Update, Bot
from telegram import ParseMode
from telegram.ext import Updater, ChatMemberHandler, MessageHandler, Filters, ContextTypes, CommandHandler

import settings
from cache_messages import MessagesCache
from handlers.commands import help_command, cancel_command, save_forwarded_message, clear_forwarded_message, \
    version_command
from settings import debug, main_users_id, chats_and_greetings, tgkey, botname, minutes_for_user_thinking


class GreetedUser:
    def __init__(self, user_id: int, greeting_bot_message: str, greeting_date: float):
        self.user_id = user_id
        self.greeting_bot_message = greeting_bot_message
        self.greeting_date = greeting_date


class UserMessage:
    def __init__(self, user_id: int, message: str):
        self.user_id = user_id
        self.message = message


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

last_greeted_user = dict()

# Contains dictionary with messages between users and bot
messages_cache = MessagesCache()


##################
# Command handlers#
##################
def remove_all_mentions(text):
    return re.sub(r'@\w+', '', text)


def answer_a_question(update):
    text_to_reply = update.message.text.replace(f'@{botname}', "")
    answer = generate_answer(update.message.from_user.id, text_to_reply)
    update.effective_chat.send_message(f'@{update.message.from_user.username} {answer}')


def reply_a_question(update):
    text_to_reply = update.message.text.replace(f'@{botname}', "")
    answer = generate_answer(update.message.from_user.id, text_to_reply)
    update.message.reply_text(text=f'@{update.message.from_user.username} {answer}',
                              reply_to_message_id=update.message.message_id)


def simple_reply(update):
    message = update.message.reply_to_message.text if update.message.reply_to_message.text is not None else update.message.text
    messages_cache.add(update.message.from_user.id, message, True)
    answer = generate_answer_raw(update.message.from_user.id, update.message.text, ignore_exceptions=False)
    update.message.reply_text(text=answer)


def answer_user_in_chat(context: ContextTypes, chat: str):
    if chat not in chats_and_greetings:
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
            update.message.chat.username not in chats_and_greetings or \
            "@" in update.message.text or \
            update.message.reply_to_message is not None or \
            last_greeted_user.get(update.message.chat.id) is None or \
            last_greeted_user[update.message.chat.id].user_id != update.message.from_user.id or \
            (time.time() - last_greeted_user[update.message.chat.id].greeting_date) / 60 > minutes_for_user_thinking:
        return False
    else:
        return True


def __available_in_group(update: Update) -> bool:
    return (update.message.chat.username is not None and update.message.chat.username in chats_and_greetings) or (
            update.message.chat.id is not None and str(update.message.chat.id) in chats_and_greetings)


def message_handler(update: Update, context: ContextTypes):
    if update.message.chat.type == "private":
        if update.message.from_user.id not in main_users_id:
            update.effective_chat.send_message(
                "Чтобы поговорить с ботом напишите в любой из чатов, где он есть, упомянув бота. например:\n\n"
                f'@{botname} Расскажи краткую историю человечества в 5 предложениях используя слова "красный" и "неудобный"',
                parse_mode=ParseMode.HTML)
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
    elif not __available_in_group(update):
        return

    if update.message.reply_to_message is not None and update.message.reply_to_message.from_user.username == botname:
        """Reply to a message."""
        comput = threading.Thread(target=simple_reply, args=(update,))
        comput.start()
    elif f'@{botname}' in update.message.text:
        comput = threading.Thread(target=reply_a_question, args=(update,))
        comput.start()
    else:
        if update.message.from_user.id in main_users_id and update.message.chat.type == "private":
            comput = threading.Thread(target=answer_a_question, args=(update,))
            comput.start()
        elif is_need_answer(update):
            comput = threading.Thread(target=simple_reply,
                                      args=(update,))
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

    if update.chat_member.chat.username not in chats_and_greetings:
        return

    """Greets new users in chats and announces when someone leaves"""
    result = extract_status_change(update.chat_member)
    if result is None:
        return

    was_member, is_member = result

    if not was_member and is_member:
        compute = threading.Thread(target=send_greet_chat_message,
                                   args=(update, chats_and_greetings.get(update.chat_member.chat.username)))
        compute.start()


def send_greet_chat_message(update, user_prompt):
    answer = generate_answer_raw(user_id=update.chat_member.new_chat_member.user.id,
                                 prompt=user_prompt.replace("{username}",
                                                            f'{update.chat_member.new_chat_member.user.first_name}'),
                                 save_in_cache=False)
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
def generate_answer(user_id, question, save_in_cache=True):
    return generate_answer_raw(user_id, question, save_in_cache)


def generate_answer_raw(user_id, prompt, save_in_cache=True, attempts=settings.total_attempts, ignore_exceptions=True):
    if save_in_cache:
        messages_cache.add(user_id, prompt, False)
    messages = messages_cache.get_formatted(user_id)
    if not save_in_cache:
        messages.append({"role": "user", "content": prompt})

    # question += f'AI:'
    user_id_str = str(user_id)
    if debug:
        print("----Start generating------")
        print("User: " + user_id_str, ", dialog:\n" + str(messages))
    try:
        response = openai.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=1500,
            user=user_id_str,
            stream=False,
            timeout=60
        )
        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo",
        #     prompt=question,
        #     temperature=0.9,
        #     max_tokens=1500,
        #     top_p=1,
        #     frequency_penalty=0.0,
        #     presence_penalty=0.6,
        #     stop=[" Human:", " AI:"],
        #     user=user_id_str
        # )
        if debug:
            print("----Response------")
            print(str(response))
        answer = response.choices[0].message.content
        messages_cache.add(user_id, answer, True)
        return answer
    except BadRequestError as e:
        print(e)
        if attempts > 0:
            if debug:
                print("Removing one old message, trying again...")
            messages_cache.remove_one_old_message(user_id)
            return generate_answer_raw(user_id, prompt, save_in_cache, attempts - 1)
        else:
            return "Мне нужно отдохнуть, я так устал..."
    except ReadTimeout as e:
        print(e)
        if ignore_exceptions:
            return "Оракул сегодня изучает числа..."
        else:
            raise e
    except RateLimitError as e:
        print(e)
        if ignore_exceptions:
            return "Так много вопросов и так мало ответов..."
        else:
            raise e


#####################
# End main functions#
#####################


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error', update)


def main():
    """Start the bot."""
    updater = Updater(tgkey)
    dp = updater.dispatcher

    settings.bot_id = updater.bot.get_me().id

    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("cancel", cancel_command))
    dp.add_handler(CommandHandler("version", version_command))
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
