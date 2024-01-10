import telebot.types
from openai import RateLimitError

from bot.constant_messages import empty_api_key_message, invalid_api_key_message, api_key_setup_successful_message, \
    api_key_update_successful_message, api_key_remove_successful_message, set_api_key_message, api_key_expired_message
from bot.constants import bot, VALID_API_KEY_TYPES, VALID_API_KEY_MODES
from bot.error_handlers import on_api_telegram_exception_429
from db.db import insert_api_key, remove_api_keys, insert_group_if_not_exists, insert_user_if_not_exists
from open_ai.common import test_openai_api_key, expire_openai_api_key


def manage_api_keys_command(message: telebot.types.Message):
    try:
        insert_user_if_not_exists(message)
        if message.chat.type == 'group' or message.chat.type == 'supergroup':
            insert_group_if_not_exists(message)
        method, key, key_type = None, None, None
        content = message.text
        if '/apikey@sv_telegram_gpt_bot' in content:
            content = content.replace('/apikey@sv_telegram_gpt_bot', '')
        elif '/apikey' in content:
            content = content.replace('/apikey', '')
        if not content:
            bot.reply_to(message, empty_api_key_message)
            return
        param_list = content.split(' ')
        for param in param_list:
            if 'help' in param:
                bot.reply_to(message, set_api_key_message)
                return
            if 'method=' in param:
                method = param.split('=')[1]
            if 'key=' in param:
                key = param.split('=')[1]
            if 'type=' in param:
                key_type = param.split('=')[1]
        invalid_params = []
        if not method or method not in VALID_API_KEY_MODES:
            invalid_params.append('method')
        if not key_type or key_type not in VALID_API_KEY_TYPES:
            invalid_params.append('type')
        if not method == 'remove':
            if not key or not test_openai_api_key(key):
                invalid_params.append('key')
        if len(invalid_params):
            bot.reply_to(message, invalid_api_key_message(', '.join(invalid_params)))
            return
        if method == 'setup':
            insert_api_key(message, key, 'openai')
            bot.reply_to(message, api_key_setup_successful_message)
        elif method == 'update':
            expire_openai_api_key(message.from_user.id)
            insert_api_key(message, key, 'openai')
            bot.reply_to(message, api_key_update_successful_message)
        elif method == 'remove':
            remove_api_keys(message.from_user.id, 'openai')
            bot.reply_to(message, api_key_remove_successful_message)
    except telebot.apihelper.ApiTelegramException:
        on_api_telegram_exception_429(manage_api_keys_command, message)
    except RateLimitError:
        bot.reply_to(message, api_key_expired_message)
    except Exception as e:
        print(str(e))


def register_api_key_command():
    bot.register_message_handler(
        manage_api_keys_command,
        commands=['apikey']
    )
