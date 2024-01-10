from __future__ import annotations

import os
import logging
import asyncio
import datetime
import json
import pprint
from dataclasses import asdict
from functools import partial

import const
import time
import traceback

from ai_tools.ai_joker import AIJoker
from ai_tools.ai_compliment import AICompliment
from ai_tools.chat_summary import ChatSummary
from ai_tools.is_relevant_group_message import IsRelevantGroupMessage
from ai_tools.exchanger_flow import ExchangerFlow
from ai_tools.is_valid_answer import IsValidAnswer
from ai_tools.news_summary import NewsSummary
from bot_logic.bot_re import BotRE
from bot_logic.bot_wrapper import BotWrapper
from lib.field_storage import FieldStorage
from ai_tools.call_manager import CallManager
from ai_tools.sentence_refactor import SentenceRefactor
from ai_tools.no_hallucinations import NoHallucinations
from ai_tools.smart_contract_analysist import SmartContractAnalysist
from ai_tools.sql_questioner import SQLQuestioner
from lib.message_limiter import MessageLimiter
from menus.mems_carousel import MemsCarousel
from menus.pins_carousel import PinsCarousel
from queues.message import Message, MessageQueue
import typing as t
import typing

from uuid import uuid4
from lib.rate_limiter import ResourceUserGroupRateLimiter

import telegram
from telegram import Update, constants, BotCommandScopeAllChatAdministrators, \
    InputMediaPhoto, Message as TelegramMessage, BotCommandScopeAllGroupChats, BotCommandScopeAllPrivateChats
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InlineQueryResultArticle
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, CallbackQueryHandler, Application, ContextTypes, CallbackContext

from pydub import AudioSegment
from ai_tools.links_replacer import LinkReplacerBabyTiger

from tools.last_messages_manager import LastMessagesManager
from db import Config, PinnedMessage
from queues.omni_send_queue import OmniSend
from queues.queue_worker import QueueWorker
from queues.triggers_queue import TriggerMessage
from tools.get_website_content import get_directory_size
from tools.knowledge_base_manager import KnowledgeBaseManager
from tools.wallet_jwt_manager import WalletJWTManager
from utils import is_group_chat, get_thread_id, message_text, wrap_with_indicator, split_into_chunks, \
    edit_message_with_retry, get_stream_cutoff_values, is_allowed, get_remaining_budget, is_admin, \
    is_within_budget_v2, \
    get_reply_to_message_id, error_handler, is_true, can_whitelist, is_mems_admin
from openai_helper import OpenAIHelper
from lib.localized import localized_text
from usage_tracker import UsageTracker, get_all_usage_trackers
from erc20prices import ERC20Prices
from dextools import DEXTools

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(CURRENT_FOLDER, 'static')


class AIImageFilter(filters.MessageFilter):
    TRIGGERS = [
        '/x',
        '/ai',
        'x',
        'ai',
        'bot',
    ]
    def filter(self, message):
        if not message.photo:
            return False
        if not message.caption:
            return False
        text = message.caption.lower()
        for trigger in self.TRIGGERS:
            if text == trigger or text.startswith(f'{trigger} '):
                logging.info(f'AIImageFilter: triggered because {message.caption=}')
                return True
        return False


class PinnedFilter(filters.MessageFilter):
    def filter(self, message: TelegramMessage):
        return message.pinned_message is not None


class PinnedByCommandFilter(filters.MessageFilter):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def filter(self, message: TelegramMessage):
        if self.config.get_last_chat_user_command(
            chat_id=message.chat_id,
            user_id=message.from_user.id,
        ) == 'pin':
            return True
        return False


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: Config):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        """
        self.config = config
        self.limiter = MessageLimiter(config)
        self.openai = OpenAIHelper(config=config)
        self.knowledge_base_manager = KnowledgeBaseManager(_config=config)
        bot_language = self.config['BOT_LANGUAGE']

        self.basic_commands = [
            BotCommand(command='help', description='How to use the Bot❓'),
            BotCommand(command='x', description='Answer questions 🤖'),
            BotCommand(command='image', description='Generate image 🎨'),
            BotCommand(command='ximage', description='Process the image 🔄🖼️'),
            BotCommand(command='etherscan', description='Smart contract analysis 🔍📜🔒'),
        ]

        self.private_commands = self.basic_commands + [
        ]

        self.group_commands = self.basic_commands + [
            BotCommand(command='realtime', description='Realtime token data 📈'),
            BotCommand(command='news_summary', description='News summary 📌'),
            BotCommand(command='chat_summary', description='Chat summary ✍️'),
            BotCommand(command='joke', description='Joke 🤣'),
            BotCommand(command='compliment', description='Compliment 🥰'),
        ]

        self.group_admin_commands = [
            BotCommand(command='setup_group', description='Setup Group'),
            BotCommand(command='noimage', description='Disable image generation'),
            BotCommand(command='yesimage', description='Enable image generation'),
            BotCommand(command='autoreply', description='Enable auto-reply'),
            BotCommand(command='noautoreply', description='Disable auto-reply'),
            BotCommand(command='mems', description='Show facts from the Knowledge base'),
            BotCommand(command='mem', description='Add fact to the Knowledge base'),
            BotCommand(command='pins', description='Show pinned messages'),
            BotCommand(command='pin', description='Add history message as pinned'),
        ] + self.group_commands

        self.disallowed_message = localized_text('disallowed', bot_language)
        self.budget_limit_message = localized_text('budget_limit', bot_language)
        self.usage = {}
        self.inline_queries_cache = {}
        self.message_queue = MessageQueue(config, 'all-messages')
        self.erc20prices = ERC20Prices(config)
        self.dextools = DEXTools(config)
        self.is_relevant_group_message = IsRelevantGroupMessage(openai=self.openai)

        if is_true(os.environ.get('RE')):
            self.field_storage = FieldStorage(config)
            self.bot_logic = BotRE(
                config=self.config,
                field_storage=self.field_storage,
                openai=self.openai,
            )
            self.bot_wrapper = BotWrapper(
                bot_logic=self.bot_logic,
                config=self.config,
                field_storage=self.field_storage,
                openai=self.openai,
            )

        self.image_gen_limiter = ResourceUserGroupRateLimiter(
            config=self.config,
            resource_name='image-gen',
            group_max_daily=150,
            group_max_hourly=10,
            group_max_minute=3,
            user_max_daily=30,
            user_max_hourly=3,
            user_max_minute=1,
        )

        self.image_ai_limiter = ResourceUserGroupRateLimiter(
            config=self.config,
            resource_name='image-ai',
            group_max_daily=300,
            group_max_hourly=100,
            group_max_minute=30,
            user_max_daily=50,
            user_max_hourly=10,
            user_max_minute=3,
        )

        self.wallet_jwt_manager = WalletJWTManager(config)

    async def handle_pinned_message_by_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        self.config.delete_last_chat_user_command(chat_id, user_id)

        if update.message.forward_from:
            sender_user = update.message.forward_from
        else:
            sender_user = update.message.from_user

        pin_message_to_save: PinnedMessage = {
            "message_id": update.message.id,
            "text": update.message.text or update.message.caption,
            "from_user": sender_user.to_dict(),
            "timestamp": None,
        }
        await self._handle_pinned_message(
            update=update,
            chat_id=chat_id,
            pin_message_to_save=pin_message_to_save,
        )
        await self.safe_reply_to_message(
            update,
            f'Historical pinned message saved. '
        )
        await context.bot.delete_message(chat_id=chat_id, message_id=update.message.id)

    async def handle_pinned_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pinned_message: TelegramMessage = update.message.pinned_message
        pin_message_to_save: PinnedMessage = {
            "message_id": pinned_message.message_id,
            "text": pinned_message.text or pinned_message.caption,
            "from_user": pinned_message.from_user.to_dict(),
            "timestamp": pinned_message.date.timestamp(),
        }
        await self._handle_pinned_message(update, pinned_message.chat_id, pin_message_to_save)

    async def _handle_pinned_message(
            self,
            update: Update,
            chat_id,
            pin_message_to_save: PinnedMessage,
    ):
        if not pin_message_to_save['text']:
            await self.safe_reply_to_message(update, 'Cannot pin empty message')
            return

        self.config.add_new_pinned_message(chat_id=chat_id, pinned_message=pin_message_to_save)
        self.config.delete_last_chat_user_command(chat_id, update.effective_user.id)

        scope_id = self.get_update_scope_id(update)
        if scope_id:  # it should not be None for set up groups
            if pin_message_to_save['timestamp']:
                _dt = datetime.datetime.fromtimestamp(pin_message_to_save['timestamp']).date().strftime("%Y-%m-%d")
            else:
                _dt = 'unknown'
            if pin_message_to_save['from_user']['username']:
                _from = pin_message_to_save['from_user']['username']
            else:
                _from = pin_message_to_save['from_user']['full_name']
            _msg = f'''
Date: {_dt}
From: {_from}

{pin_message_to_save['text']}
'''
            await self._mem(
                scope_id=scope_id,
                text=_msg,
                metadata_defaults={
                    'pin_message': pin_message_to_save,
                }
            )

    async def news_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        async def _run():
            news_summary = NewsSummary(config=self.config, openai=self.openai)
            _summary = await news_summary.get_news_summary(
                chat_id=update.effective_chat.id,
            )
            await self.safe_reply_to_message(update, _summary)
        await wrap_with_indicator(
            update,
            context,
            _run(),
            constants.ChatAction.TYPING,
        )

    async def chat_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        async def _run():
            chat_summary = ChatSummary(config=self.config, openai=self.openai, bot=self.application.bot)
            _summary = await chat_summary.get_chat_summary(
                chat_id=update.effective_chat.id,
            )
            await self.safe_reply_to_message(update, _summary)
        await wrap_with_indicator(
            update,
            context,
            _run(),
            constants.ChatAction.TYPING,
        )

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)
        commands = self.group_commands if is_group_chat(update) else self.private_commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        base = f'''
I'm AIgentX! 🤖🧠
Ask me anything related to the project and web3.
I can process text/voice/video messages, analyze smart contracts, generate images and fetch realtime token data.
'''.strip()
        help_text = (
                base +
                '\n\n' +
                '\n'.join(commands_description)
        )
        await self.safe_reply_to_message(update, help_text)

    def log(
            self,
            chat_id: int,
            action: str,  # incoming / outgoing / summary / command
            message: str,
    ):
        logging.info(f'log {chat_id=}, {action=}, {message=}')
        record_marker = '>>>'
        logs_dir = os.environ.get('CHAT_LOGS_DIR', '/tmp/chat_logs')
        os.makedirs(logs_dir, exist_ok=True)
        logs_path = os.path.join(logs_dir, f'{chat_id}.log')
        if message is None:
            logging.error(f'Empty message for {action} action')
        message_lines = message.split('\n')
        message_lines = [f'\\{line}' if line.startswith(record_marker) else line for line in
                         message_lines]  # extreme case; escape marker
        message = '\n'.join(message_lines)
        with open(logs_path, 'a') as f:
            dt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            f.write(f'{record_marker} {dt} {action}: {message}\n')

    async def log_incoming_message(self, update: Update) -> None:
        """
        Logs an incoming message.
        :param update: The incoming update
        """
        user_id = update.message.from_user.id if update.message and update.message.from_user else None
        chat_id = update.effective_chat.id
        # message_id = update.message.message_id if update.message else None
        text = update.effective_message.text if update.effective_message else None
        file_id = update.message.effective_attachment.file_id \
            if update.message and update.message.effective_attachment else None
        file_size = update.message.effective_attachment.file_size \
            if update.message and update.message.effective_attachment else None
        caption = update.effective_message.caption if update.effective_message else None
        try:
            file_name = update.message.effective_attachment.file_name if \
                update.message and update.message.effective_attachment else None
        except Exception:
            file_name = None
        try:
            if update.effective_chat.type == 'channel':
                username = update.effective_chat.title
            else:
                username = await self.get_username_by_user_id(user_id)
        except Exception as e:
            username = None
        if file_id or file_size or caption or file_name:
            self.log(
                chat_id=chat_id,
                action='incoming',
                message=f'user_id={user_id}, '
                        f'username=@{username}, '
                        f'file_id={file_id}, '
                        f'file_size={file_size}, '
                        f'caption={caption}, '
                        f'file_name={file_name}, '
                        f'text={text}'
            )
            self.message_queue.enqueue_message(
                Message(
                    channel='telegram',
                    user_id=str(user_id),
                    username=username,
                    chat_id=str(chat_id),
                    content=f'filename: {file_name}, caption: {caption}',
                )
            )
        else:
            self.log(
                chat_id=chat_id,
                action='incoming',
                message=f'user_id={user_id}, username=@{username}, text={text}'
            )
            self.message_queue.enqueue_message(
                Message(
                    channel='telegram',
                    user_id=str(user_id),
                    username=username,
                    chat_id=str(chat_id),
                    content=text,
                )
            )

    def log_conversation(self, chat_id: int) -> None:
        conversation = self.openai.conversation(chat_id)
        self.log(
            chat_id=chat_id,
            action='conversation',
            message=json.dumps(conversation, ensure_ascii=False, indent=4),
        )

    def log_outcoming_message(self, chat_id: int, reply: str) -> None:
        self.log(chat_id, 'reply', reply)

    def log_command(self, chat_id: int, update):
        if update.effective_message is None:
            message = 'None'
        elif update.effective_message.text:
            message = update.effective_message.text
        elif update.effective_message.caption:
            message = update.effective_message.caption
        else:
            message = 'None'
        self.log(chat_id, 'command', message)

    def _get_logs(
            self,
            chat_id: int,
            start_dt: datetime.datetime,
            limit: int = 10
    ) -> list[str]:
        logs_dir = os.environ.get('CHAT_LOGS_DIR', '/tmp/chat_logs')
        logs_path = os.path.join(logs_dir, f'{chat_id}.log')
        if not os.path.exists(logs_path):
            return []
        with open(logs_path, 'r') as f:
            lines = f.readlines()
        result = []
        messages_count = 0

        def format_conversation(conversation_json):
            items = []
            for item in conversation_json:
                item = f'''{{
    "role": "{item['role']}",
    "content": """{item['content']}"""
}}'''
                items.append(item)
            result = '[' + ',\n'.join(items) + ']'
            return result

        for line in lines:
            line = line.strip()
            if line.startswith('>>>'):
                if messages_count >= limit:
                    break
                dt = datetime.datetime.strptime(line[4:23], '%Y-%m-%dT%H:%M:%S')
                if start_dt is None or dt >= start_dt:
                    after_dt = line[24:]
                    # logging.info(f'after_dt={after_dt[:100]}')
                    if after_dt.startswith('conversation: '):
                        conversation_json = json.loads(after_dt[len('conversation: '):])
                        line = line[:24] + 'conversation: ' + format_conversation(conversation_json)
                    result.append(line)
                    messages_count += 1
            elif messages_count:
                result.append(line)  # append message lines
        return result

    async def get_logs(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        text = update.effective_message.text
        parts = text.split(' ')
        assert len(parts) == 4, 'Usage: /get_logs <chat_id> <start_dt> <limit>'
        assert parts[0] == '/get_logs', 'Usage: /get_logs <chat_id> <start_dt> <limit>'
        chat_id = int(parts[1])
        if parts[2] == 'all':
            start_dt = None
        else:
            start_dt = datetime.datetime.strptime(parts[2], '%Y-%m-%dT%H:%M:%S')
        limit = int(parts[3])
        logs = self._get_logs(chat_id, start_dt, limit)
        reply = '\n'.join(logs)
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def delete(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        key = parts[1]

        self.config.delete(key)
        reply = f'/delete {key}'

        logging.info(f'/delete {key}')
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def get_keys_by_pattern(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_send_message_to_thread(update, "You are not allowed to use this command.", technical=True)
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        key = parts[1]

        assert command == '/get_keys_by_pattern'
        result = self.config.get_keys_by_pattern(key)
        reply = f'/get_keys_by_pattern {key}:\n' + '\n'.join(result)

        logging.info(f'/get_keys_by_pattern result={result}')
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def delete_keys_by_pattern(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        pattern = parts[1]
        assert command == '/delete_keys_by_pattern'

        if pattern == '*':
            reply = 'Cannot delete all keys.'
        else:
            result = self.config.get_keys_by_pattern(pattern)
            for key in result:
                self.config.delete(key)
            reply = f'/delete_keys_by_pattern {pattern}:\n' + '\n'.join(result)

        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def get(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        key = parts[1]

        assert command == '/get'
        reply = f'{key}={self.config[key]}'

        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def get_chat_id(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.effective_message.from_user.id
        chat_id = update.effective_chat.id
        reply = f'chat_id={chat_id}, {user_id=}, chat_type={update.effective_chat.type}'
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=reply,
        )

    async def add_chat_id_to_REPLIES(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        old = self.config['REPLIES_CHAT_IDS'] or ''
        if old:
            new = str(old) + ',' + str(chat_id)
        else:
            new = str(chat_id)
        new = ','.join(set(new.split(',')))
        self.config['REPLIES_CHAT_IDS'] = new
        reply = f'add_chat_id_to_REPLIES:\nchat_id={chat_id}\nREPLIES_CHAT_IDS={new}'
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def add_chat_id_to_KWARGS(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        old = self.config['OPENAI_KWARGS_CHAT_IDS'] or ''
        if old:
            new = str(old) + ',' + str(chat_id)
        else:
            new = str(chat_id)
        new = ','.join(set(new.split(',')))
        self.config['OPENAI_KWARGS_CHAT_IDS'] = new
        reply = f'add_chat_id_to_KWARGS:\nchat_id={chat_id}\nOPENAI_KWARGS_CHAT_IDS={new}'
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def add_chat_id_to_BUDGETS(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        old = self.config['BUDGET_CHAT_IDS'] or ''
        if old:
            new = str(old) + ',' + str(chat_id)
        else:
            new = str(chat_id)
        new = ','.join(set(new.split(',')))
        self.config['BUDGET_CHAT_IDS'] = new
        reply = f'add_chat_id_to_BUDGETS:\nchat_id={chat_id}\nBUDGET_CHAT_IDS={new}'
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def add_chat_id_to_TRIGGERS(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        old = self.config['TRIGGER_CHAT_IDS'] or ''
        if old:
            new = str(old) + ',' + str(chat_id)
        else:
            new = str(chat_id)
        new = ','.join(set(new.split(',')))
        self.config['TRIGGER_CHAT_IDS'] = new
        reply = f'add_chat_id_to_TRIGGERS:\nchat_id={chat_id}\nTRIGGER_CHAT_IDS={new}'
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def search_vectors(self, update: Update, _: ContextTypes.DEFAULT_TYPE, to_file=False) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        if text.startswith('/search_vectors '):
            text = text[len('/search_vectors '):]
        elif text.startswith('/search_vectorsf '):
            text = text[len('/search_vectorsf '):]
        else:
            raise ValueError(f'Unknown command: {text}')
        parts = text.split(' ')
        k = int(parts[0])
        min_score = float(parts[1])
        scope_id = parts[2]
        if scope_id.lower() == 'none':
            scope_id = None
        text = ' '.join(parts[3:])
        helpers = await self.openai.semantic_db.search(
            text,
            top_k=k,
            min_score=min_score,
            scope_id=scope_id
        )
        reply = f"""Search results for "{text}":
{pprint.pformat(asdict(helpers))}"""
        logging.info(f'{reply=}')

        if to_file:
            await self.safe_send_file(
                update=update,
                file_content=reply,
                filename=f'search_vectors.txt',
                caption=f'Search results for "{text}":',
            )
        else:
            await self.safe_send_message_to_thread(update, reply, technical=True)

    async def get_scope_id(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return
        scope_id = self.get_update_scope_id(update)
        reply = f'scope_id={scope_id}'
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def pins(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        username = update.message.from_user.username

        parts = update.message.text.split(' ')
        logging.info(f'pins {parts=}')

        _is_admin = is_admin(self.config, user_id, username=username)
        # _is_mems_admin = is_mems_admin(self.config, user_id, username=username)
        _is_pins_admin = False

        if update.effective_chat.type == 'private':
            if not _is_admin and not _is_pins_admin:
                await self.safe_reply_to_message(update, "You are not allowed to use this command.")
                return
            if len(parts) == 2:
                scope_id = parts[1]
            else:
                scope_id = self.get_update_scope_id(update)
        else:  # group
            is_group_admin = await self.check_if_group_admin(chat_id=chat_id, user_id=user_id)
            if not _is_admin and not is_group_admin and not _is_pins_admin:
                await self.safe_reply_to_message(update, "You are not allowed to use this command.")
                return

            if len(parts) == 1:
                scope_id = self.get_update_scope_id(update)
                if scope_id is None and not _is_pins_admin and not _is_admin:
                    await self.safe_reply_to_message(update, "Setup is not finished for this group.")
                    return
            else:
                if not _is_admin:
                    await self.safe_reply_to_message(update, "You are not allowed to use this command.")
                    return
                scope_id = parts[1]

        logging.info(f'pins {scope_id=}')
        carousel = PinsCarousel(
            config=self.config,
            chat_id=chat_id,
            semantic_db_client=self.openai.semantic_db,
            send_message_to_chat_id=self.send_message_to_chat_id,
        )
        await carousel.send(update, _)

    async def mems(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        username = update.message.from_user.username

        parts = update.message.text.split(' ')
        logging.info(f'mems {parts=}')

        _is_admin = is_admin(self.config, user_id, username=username)
        _is_mems_admin = is_mems_admin(self.config, user_id, username=username)

        if update.effective_chat.type == 'private':
            if not _is_admin and not _is_mems_admin:
                await self.safe_reply_to_message(update, "You are not allowed to use this command.")
                return
            if len(parts) == 2:
                scope_id = parts[1]
            else:
                scope_id = self.get_update_scope_id(update)
        else:  # group
            is_group_admin = await self.check_if_group_admin(chat_id=chat_id, user_id=user_id)
            if not _is_admin and not is_group_admin and not _is_mems_admin:
                await self.safe_reply_to_message(update, "You are not allowed to use this command.")
                return

            if len(parts) == 1:
                scope_id = self.get_update_scope_id(update)  # could be None
                if scope_id is None and not _is_mems_admin and not _is_admin:
                    await self.safe_reply_to_message(update, "Setup is not finished for this group.")
                    return
            else:
                if not _is_admin:
                    await self.safe_reply_to_message(update, "You are not allowed to use this command.")
                    return
                scope_id = parts[1]

        logging.info(f'mems {scope_id=}')
        carousel = MemsCarousel(
            scope_id=scope_id,
            semantic_db_client=self.openai.semantic_db,
            send_message_to_chat_id=self.send_message_to_chat_id,
        )
        await carousel.send(update, _)

    async def kb(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id

        parts = update.message.text.split(' ')
        logging.info(f'mems {parts=}')
        if len(parts) == 1:
            scope_id = self.get_update_scope_id(update)
        else:
            scope_id = parts[1]

        logging.info(f'mems {scope_id=}')
        helpers = await self.openai.semantic_db.get(scope_id=scope_id)

        await self.safe_send_file(
            file_content=json.dumps(asdict(helpers), indent=4),
            filename=f'mems_scope_{scope_id}.json',
            update=update,
            caption=f'Found {len(helpers.documents)} mems for scope: {scope_id}',
        )

    async def set_from_file(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        logging.info(f'set_from_file')
        self.log_command(update.effective_chat.id, update)

        async def warn_msg():
            logging.warning(f'incorrect command format')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text="Incorrect command format, "
                     "should be "
                     "\"/set_from_file key1 key2 key3\" + attached text file."
                # parse_mode=constants.ParseMode.MARKDOWN
            )

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.caption
        parts = text.split(' ')
        command = parts[0]
        keys = {_ for _ in parts[1:]}
        if not keys:
            keys = None

        if not update.message.document:
            await warn_msg()
            return

        document = update.message.document
        logging.info(f"Received document with file id: {document.file_id}")

        file = await self.safe_telegram_request(self.application.bot.getFile, document.file_id)
        filename = '/tmp/received_text_file.txt'
        await file.download_to_drive(filename)
        with open(filename, 'r') as f:
            contents = f.read()
        os.remove(filename)

        reply = []
        key = ''
        value = ''

        async def set_key(key, value):
            logging.info(f'__ set_key {key=} {value=}')
            if key and (keys is None or key in keys):
                logging.info(f'SET {key=}, {value=}')
                if keys:
                    keys.remove(key)

                value = value.strip()

                if key == 'RAW_QUESTIONS':
                    await self.just_set_questions(value)
                    reply.append(f'LOAD RAW_QUESTIONS')
                    self.config[key] = value
                else:
                    self.config[key] = value
                    reply.append(f'{key}={value}')

        is_multiline = False
        for line in contents.split('\n'):
            line = line.strip()

            if line == '====':
                # logging.info(f'__ ====')

                await set_key(key, value)
                key = ''
                value = ''
                is_multiline = False
                continue
            part0 = line.split(' ')[0]
            if len(part0.split('=')) == 2 and not is_multiline:
                # logging.info(f'__ 2222')

                await set_key(key, value)
                key, value = line.split('=')
                continue
            else:
                # logging.info(f'__ 3333')
                if key:
                    is_multiline = True
                    value += '\n' + line

        await set_key(key, value)
        del key, value

        reply = '\n'.join(reply)
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def check_if_group_admin(
            self,
            chat_id,
            user_id,
            check_anon_admins=False,
    ):
        if check_anon_admins and user_id in [777000, 1087968824]:
            return True

        if is_admin(self.config, user_id):
            logging.info(f'{user_id=} is admin of {chat_id=}, because is_admin')
            return True

        chat = await self.application.bot.get_chat(chat_id)
        try:
            if chat.type == "group" and chat.all_members_are_administrators:
                logging.info(
                    f'{user_id=} is admin of {chat_id=}, because {chat.type=} and {chat.all_members_are_administrators=}')
                return True
        except Exception as exc:
            logging.exception(f'check_if_group_admin failed: {exc}')
            pass
        chat_admins = await self.application.bot.get_chat_administrators(chat_id)
        user_ids = [admin.user.id for admin in chat_admins]
        status = user_id in user_ids
        if status:
            logging.info(f'{user_id=} is admin of {chat_id=}, because {chat.type=} and {chat_admins=}')
            return True
        else:
            logging.info(f'{user_id=} is NOT admin of {chat_id=}, because {chat.type=} and {chat_admins=}')
            return False

    async def verify(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)
        user_id = update.message.from_user.id

        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not Bot admin.")
            return

        if not self.check_if_group_admin(update.effective_chat.id, user_id):
            await self.safe_reply_to_message(update, "You are not admin of this group.")
            return

        group_id = update.effective_chat.id
        link = f'https://shy-wave-3009.on.fleek.co/?groupId={group_id}'
        reply = f'🛡️ Approve AIX ownership via this link: {link}'
        await self.safe_send_message_to_thread(update, reply)

    async def verify_reply(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)
        user_id = update.message.from_user.id

        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not Bot admin.")
            return

        if not self.check_if_group_admin(update.effective_chat.id, user_id):
            await self.safe_reply_to_message(update, "You are not admin of this group.")
            return

        reply = f'✅ Approved! Now call "/setup" command to initialize your AI bot. 🤖'
        await self.safe_send_message_to_thread(update, reply)

    async def fake_setup(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)
        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not Bot admin.")
            return

        text = update.message.text
        parts = text.split(' ')
        website = parts[1]

        await self.safe_send_message_to_thread(update, f'⏳ Setup {website} started, wait ~30 seconds')
        await asyncio.sleep(5)
        await self.safe_send_message_to_thread(update, f'🚀 Setup {website} done, 10 pages processed')

    async def get_group_name(self, group_id):
        chat = await self.application.bot.get_chat(chat_id=group_id)
        return chat.title

    def gen_reply_markup_ton_of_voice(self, group_id):
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("👔💼 Formal", callback_data=f"ton_of_voice:{group_id}:formal")],
            [InlineKeyboardButton("😊🤝 Friendly", callback_data=f"ton_of_voice:{group_id}:friendly")],
            [InlineKeyboardButton("🚀🌑 Enthusiastic", callback_data=f"ton_of_voice:{group_id}:enthusiastic")],
            [InlineKeyboardButton("🧠🖥️ Technical", callback_data=f"ton_of_voice:{group_id}:technical")],
            [InlineKeyboardButton("😜️🎈 Humorous", callback_data=f"ton_of_voice:{group_id}:humorous")],
        ])
        return reply_markup

    def gen_reply_markup(self, group_id):
        autoreply = self.config[f'group_settings:{group_id}:autoreply']
        if not autoreply:
            autoreply_text = '🤔 Auto-Reply'
        elif is_true(autoreply):
            autoreply_text = '✅ Auto-Reply'
        else:
            autoreply_text = '⛔ Auto-Reply'

        if is_true(self.config[f'NONCRYPTO_MULTI']):
            company_info = self.config[f'group_settings:{group_id}:company_info']
            website = self.config[f'group_settings:{group_id}:website']
            ton_of_voice = self.config[f'group_settings:{group_id}:ton_of_voice']

            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton(('✅' if company_info else 'ℹ') + "️ set company information",
                                      callback_data=f"company_info:{group_id}")],
                [InlineKeyboardButton(('✅' if website else '🌐') + " set website",
                                      callback_data=f"website:{group_id}")],
                [InlineKeyboardButton(autoreply_text, callback_data=f"autoreply:{group_id}")],
                [InlineKeyboardButton(('✅' if ton_of_voice else '🗣') + " set tone of voice",
                                      callback_data=f"select_ton_of_voice:{group_id}")],
                [InlineKeyboardButton('✖️ Cancel', callback_data=f'cancel:{group_id}')],
                [InlineKeyboardButton('⛔ Delete', callback_data=f'delete:{group_id}')],
                [InlineKeyboardButton('🎉 Finish', callback_data=f'finish:{group_id}')],
            ])
        else:
            verified = self.config[f'group_settings:{group_id}:verified']
            company_info = self.config[f'group_settings:{group_id}:company_info']
            website = self.config[f'group_settings:{group_id}:website']
            ton_of_voice = self.config[f'group_settings:{group_id}:ton_of_voice']
            coingecko = self.config[f'group_settings:{group_id}:coingecko']
            token_info = self.config[f'group_settings:{group_id}:token_info']
            how_to_buy = self.config[f'group_settings:{group_id}:how_to_buy']
            contracts = self.config[f'group_settings:{group_id}:contracts']
            dex_tools_link = self.config[f'group_settings:{group_id}:dex_tools_link']

            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton(
                    ('✅' if verified else '🔐') + " verify by wallet",
                    callback_data=f"verified:{group_id}")],


                [
                    InlineKeyboardButton(
                        "ℹ️ Set Company Info",
                        callback_data=f"company_info:{group_id}"),
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"company_info:{group_id}:get")]
                    if company_info else []
                ),


                [
                    InlineKeyboardButton(
                        "📈 Set CoinGecko",
                        callback_data=f"coingecko:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"coingecko:{group_id}:get")]
                    if coingecko else []
                ),


                [
                    InlineKeyboardButton(
                        "🔗 Set DexTools",
                        callback_data=f"dex_tools_link:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"dex_tools_link:{group_id}:get")]
                    if dex_tools_link else []
                ),


                [
                    InlineKeyboardButton(
                        "ℹ️ Set Token Info",
                        callback_data=f"token_info:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"token_info:{group_id}:get")]
                    if token_info else []
                ),


                [
                    InlineKeyboardButton(
                        "ℹ️ Set How To Buy",
                        callback_data=f"how_to_buy:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"how_to_buy:{group_id}:get")]
                    if how_to_buy else []
                ),


                [
                    InlineKeyboardButton(
                        "📜 Set Contracts",
                        callback_data=f"contracts:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"contracts:{group_id}:get")]
                    if contracts else []
                ),


                [
                    InlineKeyboardButton(
                        "🌐 Set Websites",
                        callback_data=f"website:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"website:{group_id}:get")]
                    if website else []
                ),


                [
                    InlineKeyboardButton(autoreply_text, callback_data=f"autoreply:{group_id}"),
                    InlineKeyboardButton("🟢 Enable", callback_data=f"autoreply:{group_id}:enable"),
                    InlineKeyboardButton("🔴 Disable", callback_data=f"autoreply:{group_id}:disable"),
                ],


                [
                    InlineKeyboardButton(
                        "🗣 Set Style",
                        callback_data=f"select_ton_of_voice:{group_id}")
                ] + (
                    [InlineKeyboardButton("✅ View", callback_data=f"select_ton_of_voice:{group_id}:get")]
                    if ton_of_voice else []
                ),


                [InlineKeyboardButton('✖️ Cancel', callback_data=f'cancel:{group_id}')],
                [InlineKeyboardButton('⛔ Delete', callback_data=f'delete:{group_id}')],
                [InlineKeyboardButton('🎉 Finish', callback_data=f'finish:{group_id}')],
            ])
        return reply_markup

    async def setup_group(self, update: Update, context: CallbackContext) -> None:
        chat_id = update.message.chat_id
        user_id = update.message.from_user.id
        bot = context.bot

        if is_true(self.config.get_setup_group_disabled()):
            await self.safe_reply_to_message(update, "Setup is disabled.")
            return

        chat_type = update.message.chat.type
        if chat_type not in ['group', 'supergroup']:
            await self.safe_reply_to_message(update, "This command is available for groups only.")
            return

        is_group_admin = await self.check_if_group_admin(chat_id, user_id)
        if not is_group_admin:
            await self.safe_reply_to_message(update, "You are not admin of this group.")
            return

        if not is_admin(self.config, user_id):
            groups_count = self.config.get(f'aix:groups_count:{user_id}') or 0
            groups_count += 1
            logging.info(f'{user_id=} {groups_count=}')
            if groups_count > 30:
                await self.safe_reply_to_message(update, "You can setup only 30 groups.")
                return
            else:
                self.config[f'aix:groups_count:{user_id}'] = groups_count

        chat_admins = await bot.get_chat_administrators(chat_id)
        if bot.id not in [admin.user.id for admin in chat_admins]:
            await self.send_media_group(
                update,
                [
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'admin1.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'admin2.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'admin3.png'), 'rb')),
                ],
                caption="Add me as admin please.",
            )
            await self.safe_reply_to_message(update, "I'm not an admin of this group.")
            return

        group_id = update.message.chat.id
        group_name = update.message.chat.title or "No title"
        try:
            reply_markup = self.gen_reply_markup(group_id)
            await self.send_message_to_chat_id(
                user_id,
                f"I'm ready to setup your group, '{group_name}' #{group_id}",
                reply_markup=reply_markup,
            )
            self.config[f'setup:{user_id}:group_id'] = group_id
            self.config[f'setup:{user_id}:step'] = 'initial'
            self.config[f'group_settings:{group_id}:group_name'] = group_name
            self.config.add_user_chat(user_id, group_id)
            await self.safe_reply_to_message(update, "OK, please check you DM to continue.")
        except telegram.error.Forbidden as exc:
            logging.warning(f'failed to send message: {type(exc)=} {exc}')
            with open(os.path.join(STATIC_FOLDER, f'start_dm_step1.png'), "rb") as photo:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                )
            with open(os.path.join(STATIC_FOLDER, f'start_dm_step2.png'), "rb") as photo:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption="Start a private chat with me."
                )
            await self.safe_reply_to_message(update, "Please start a private chat with me before setup.")

    async def pause(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        if not is_admin(self.config, update.message.from_user.id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        self.config['PAUSE'] = 'true'
        await self.safe_send_message_to_thread(update, 'Paused', technical=True)

    async def unpause(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        if not is_admin(self.config, update.message.from_user.id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        self.config['PAUSE'] = 'false'
        await self.safe_send_message_to_thread(update, 'Unpaused', technical=True)

    async def noimage(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        _is_admin = await self.check_if_group_admin(
            update.effective_chat.id,
            update.message.from_user.id,
            check_anon_admins=True,
        )

        if not _is_admin:
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        key = f'chat:{chat_id}:noimage'
        self.config[key] = 'true'

        await self.safe_send_message_to_thread(update, 'Image generation was disabled', technical=True)

    async def yesimage(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        _is_admin = await self.check_if_group_admin(
            update.effective_chat.id,
            update.message.from_user.id,
            check_anon_admins=True,
        )

        if not _is_admin:
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        key = f'chat:{chat_id}:noimage'
        del self.config[key]

        await self.safe_send_message_to_thread(update, 'Image generation was enabled', technical=True)

    async def autoreply(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        _is_admin = await self.check_if_group_admin(
            update.effective_chat.id,
            update.message.from_user.id,
            check_anon_admins=True,
        )

        if not _is_admin:
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        key = f'group_settings:{chat_id}:autoreply'
        self.config[key] = 'true'

        await self.safe_send_message_to_thread(update, 'Autoreply was enabled', technical=True)

    async def noautoreply(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        _is_admin = await self.check_if_group_admin(
            update.effective_chat.id,
            update.message.from_user.id,
            check_anon_admins=True,
        )

        if not _is_admin:
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        chat_id = update.effective_chat.id
        key = f'group_settings:{chat_id}:autoreply'
        self.config[key] = 'false'

        await self.safe_send_message_to_thread(update, 'Autoreply was disabled', technical=True)

    async def set_wallet(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        async def warn_msg():
            logging.warning(f'incorrect command format')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text="Incorrect command format, "
                     "should be "
                     "\"/set_wallet {wallet} {TOKEN}\""
            )

        user_id = update.message.from_user.id
        username = update.message.from_user.username

        text = update.message.text
        if text:
            text = text.strip()
        parts = text.split(' ')
        if len(parts) != 3:
            logging.warning(f'bad {parts=}')
            await warn_msg()
            return

        try:
            wallet: str = parts[1].lower()
            assert wallet.startswith('0x')
            assert set(wallet[2:]) < set('0123456789abcdef')
            assert len(wallet) == 42
            token: str = parts[2]
        except Exception as exc:
            logging.exception(f'bad {parts=}')
            await warn_msg()
            return

        _status, _wallet = self.wallet_jwt_manager.verify_token(token)
        if not _status:
            await self.safe_reply_to_message(update, "ERROR: Invalid token.")
            return
        if _wallet.lower() != wallet.lower():
            await self.safe_reply_to_message(update, "ERROR: Wallet mismatch.")
            return

        self.config.reset_user_wallet(user_id=user_id, wallet=wallet)
        self.config.set_user_username(user_id=user_id, username=username)

        reply = f'Success!\nWallet: {wallet}\nConnected to User: @{username} #{user_id}'
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def set(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        async def warn_msg():
            logging.warning(f'incorrect command format')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text="Incorrect command format, "
                     "should be "
                     "\"/set key\" + attached text file. "
                     "Or use \"/set KEY VALUE\" or \"/set KEY=VALUE\" with no file attached.",
                # parse_mode=constants.ParseMode.MARKDOWN
            )

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        key = parts[1]

        value_parts = parts[2:] if len(parts) > 2 else []
        if '=' in key:
            if key.count('=') != 1:
                await warn_msg()
                return
            value_parts = [key.split('=')[1]] + value_parts
            key = key.split('=')[0]
        value = ' '.join(value_parts)

        assert command == '/set'
        reply = f'SET {key}={value}'
        logging.info(reply)
        self.config[key] = value

        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def whitelist(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        async def warn_msg():
            logging.warning(f'incorrect command format')
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text="Incorrect command format"
            )

        user_id = update.message.from_user.id
        username = update.message.from_user.username
        if not is_admin(self.config, user_id) and not can_whitelist(self.config, user_id, username=username):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        address = parts[1]

        old = self.config['WHITELIST'] or ''
        if old:
            new = str(old) + ',' + str(address.lower())
        else:
            new = str(address.lower())
        new = ','.join(set(new.split(',')))
        self.config['WHITELIST'] = new

        reply = f'WHITELISTED ADDRESS: {address}\n\nCurrent WHITELIST: {new}'
        logging.info(reply)

        await self.safe_send_message_to_thread(update, reply, technical=True)

    def get_today_budget_message(self):
        today_budget = self.openai.budget_client.get_total_client_budget_daily(client=os.environ.get('CLIENT'))
        top_users = self.openai.budget_client.get_top_chats_daily()
        _top_users = []
        for _ in top_users:
            usage = _['usage']
            chat_id = _['chat_id']
            key_chat = f'name:chat:{chat_id}'
            key_name = f'name:user:{chat_id}'
            if self.config.get(key_name):
                name = '@' + str(self.config.get(key_name)) + ' #' + str(chat_id)
            elif self.config.get(key_chat):
                name = 'group: ' + str(self.config.get(key_chat)) + ' #' + str(chat_id)
            else:
                name = '#' + str(chat_id)
            _top_users.append(f'{name}: ${usage}')
        top_users = '\n'.join(_top_users)

        reply = f"""
Today's budget: ${today_budget}
Top chats:
{top_users}
""".strip()
        return reply

    async def today_budget(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)
        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]

        reply = self.get_today_budget_message()
        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def reset_budgets(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        self.message_queue.enqueue_message(Message(
            channel='telegram',
            user_id=str(update.message.from_user.id),
            username=update.message.from_user.username,
            chat_id=str(update.effective_chat.id),
            content=update.message.text,
        ))

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        self.openai.budget_client.clean()

        reply = 'All budgets reset.'
        await self.safe_send_message_to_thread(update, reply)

    # async def pause_instagram_chat(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    #     self.log_command(update.effective_chat.id, update)
    #     user_id = update.message.from_user.id
    #     if not is_admin(self.config, user_id):
    #         await self.safe_reply_to_message(update, "You are not allowed to use this command.")
    #         return
    #
    #     text = update.message.text
    #     parts = text.split(' ')
    #     username = parts[1]
    #     username = '@' + username if not username.startswith('@') else username
    #     if len(parts) == 3:
    #         expiry = int(parts[2])
    #         reply = f'pause {username} for {expiry} seconds'
    #     else:
    #         expiry = None
    #         reply = f'pause {username} forever'
    #
    #     logging.info(reply)
    #
    #     if expiry:
    #         self.config.setex(f'instagram:{username}', True, expiry)
    #     else:
    #         self.config.set(f'instagram:{username}', True)
    #
    #     await update.effective_message.reply_text(
    #         message_thread_id=get_thread_id(update),
    #         text=reply,
    #     )
    #
    # async def unpause_instagram_chat(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    #     self.log_command(update.effective_chat.id, update)
    #     user_id = update.message.from_user.id
    #     if not is_admin(self.config, user_id):
    #         await self.safe_reply_to_message(update, "You are not allowed to use this command.")
    #         return
    #
    #     text = update.message.text
    #     parts = text.split(' ')
    #     username = parts[1]
    #     username = '@' + username if not username.startswith('@') else username
    #     reply = f'unpause {username}'
    #
    #     logging.info(reply)
    #     self.config.delete(f'instagram:{username}')
    #     await update.effective_message.reply_text(
    #         message_thread_id=get_thread_id(update),
    #         text=reply,
    #     )

    async def email(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        # user_id = update.message.from_user.id
        # if not is_admin(self.config, user_id):
        #     await self.safe_reply_to_message(update, "You are not allowed to use this command.")
        #     return

        text = update.message.text
        parts = text.split('\n')
        command = parts[0].strip()
        to = parts[1]
        subject = parts[2]
        body = '\n'.join(parts[3:])

        assert command == '/email'
        email_json = {
            "to": to,
            "subject": subject,
            "body": body,
        }
        email_json_str = json.dumps(email_json, indent=4, ensure_ascii=False)
        self.config.r.lpush('email_queue', email_json_str)
        reply = f'OK:\n{email_json_str}'

        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def exception(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        _command = parts[0]
        name = parts[1]
        value = ' '.join(parts[2:])
        TestException = type(
            f"TestException{name}",
            (Exception,),
            {"counter": 0,
             "__init__": lambda self, message: setattr(self, "message", message),
             "__str__": lambda self: self.message}
        )

        exc = TestException(value)
        reply = f'exception {type(exc)=} {exc=}'

        try:
            raise exc
        except Exception as e:
            logging.exception(e)

        await self.safe_send_message_to_thread(update, reply, technical=True)

    #     async def ai(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    #         chat_id = update.effective_chat.id
    #         text = update.message.text
    #         parts = text.split(' ')
    #         command = parts[0]
    #         code = ' '.join(parts[1:])
    #         self.config[f'scope_id:{chat_id}'] = code.strip()
    #
    #         reset_content = message_text(update.message)
    #         await self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
    #
    #         scope_id = code.strip()
    #
    #         website = self.config[f'scope:{scope_id}:WEBSITE']
    #         if not website:
    #             out = f"""
    # You have entered an invalid code.
    #
    # 💡You can find your activation code in the message you received from us.
    # 🧑🏽‍💻 Support email — vadim@salesbotai.co
    # """.strip()
    #             await self.safe_send_message(update, out)
    #             return
    #         elif not (await self.openai.semantic_db.scope_exists(scope_id)):
    #             out = f"""
    # Your demo is activating, please wait ~30 seconds.
    # """.strip()
    #             await self.safe_send_message(update, out)
    #             await self._add_website(website=website, scope_id=scope_id, verbose=False, force_load_to_db=True, update=None)
    #
    #         NAME = self.config[f'scope:{scope_id}:NAME']
    #         WEBSITE = self.config[f'scope:{scope_id}:WEBSITE']
    #
    #         reply = f'''
    # Hello, I am an AI Sales Agent of the company "{NAME}", {WEBSITE}, how can I help you today?
    #
    # You can text me in any language
    # 🇬🇧🇨🇳🇪🇸🇮🇳🇷🇺🇺🇦🇦🇪🇫🇷🇯🇵🇩🇪🇹🇷🇰🇷🇮🇱🌍...
    #
    # 💡Examples of questions you can ask me:
    #
    # – Tell me more about your company.
    # – Help me to choose the best product/service option for me.
    # – How can I make a purchase?
    # – What can I ask you?
    # – ... whatever you want.
    # '''.strip()
    #
    #         await self.openai.reset_chat_history(chat_id=chat_id, scope_id=scope_id)
    #         self.openai._add_to_history(
    #             chat_id=chat_id,
    #             role='assistant',
    #             content=reply
    #         )
    #
    #         for index, transcript_chunk in enumerate(split_into_chunks('Congratulations! Your demo is activated. 🤖')):
    #             await update.effective_message.reply_text(
    #                 message_thread_id=get_thread_id(update),
    #                 text=transcript_chunk,
    #             )
    #
    #         for index, transcript_chunk in enumerate(split_into_chunks(reply)):
    #             await update.effective_message.reply_text(
    #                 message_thread_id=get_thread_id(update),
    #                 text=transcript_chunk,
    #             )

    async def x(self, update: Update, context) -> None:
        await self.prompt(update, context)

    async def pin(self, update: Update, context) -> None:
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        self.config.set_last_chat_user_command(chat_id, user_id, 'pin')
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton(
                '✖️ Cancel',
                callback_data=f'cancel-pin:{chat_id}:{user_id}',
            )],
        ])
        await self.safe_reply_to_message(
            update,
            'Forward a historical message to this chat, I will remember it as pinned.',
            reply_markup=reply_markup,
        )

    async def joke(self, update: Update, context) -> None:
        async def _run():
            chat_id = update.effective_chat.id
            user_id = update.message.from_user.id
            conversation = ''
            if update.message.reply_to_message:
                reply_to_message = self.last_messages_manager.get_message_by_ref(update.message.reply_to_message)
                if not reply_to_message:
                    await self.safe_reply_to_message(
                        update,
                        'I cannot access the message you forwarded. Ha-ha.',
                    )
                    return
                text = reply_to_message.text or reply_to_message.caption
                if reply_to_message.from_user.username:
                    _from = '@' + reply_to_message.from_user.username
                else:
                    _from = reply_to_message.from_user.full_name
                conversation = f'{_from}:\n{text}\n----\n'

            text = update.message.text
            parts = text.split(' ')
            text = ' '.join(parts[1:])
            if text:
                if update.message.from_user.username:
                    _from = '@' + update.message.from_user.username
                else:
                    _from = update.message.from_user.full_name
                conversation += f'{_from}:\n{text}'

            conversation = conversation.strip()
            joker = AIJoker(
                config=self.config,
                openai=self.openai,
            )
            joke = await joker.get_joke(chat_id=chat_id, conversation=conversation)
            await self.safe_reply_to_message(
                update,
                joke,
            )

        await wrap_with_indicator(
            update=update,
            context=context,
            coroutine=_run(),
            chat_action=constants.ChatAction.TYPING,
        )

    async def compliment(self, update: Update, context) -> None:
        async def _run():
            chat_id = update.effective_chat.id
            user_id = update.message.from_user.id
            conversation = ''

            if update.message.from_user.username:
                _sender = '@' + update.message.from_user.username
            else:
                _sender = update.message.from_user.full_name

            if update.message.reply_to_message:
                reply_to_message = self.last_messages_manager.get_message_by_ref(update.message.reply_to_message)
                if not reply_to_message:
                    await self.safe_reply_to_message(
                        update,
                        'I cannot access the message you forwarded. Ha-ha.',
                    )
                    return
                text = reply_to_message.text or reply_to_message.caption
                if reply_to_message.from_user.username:
                    _from = '@' + reply_to_message.from_user.username
                else:
                    _from = reply_to_message.from_user.full_name
                conversation = f'{_from}:\n{text}\n----\n'

            text = update.message.text
            parts = text.split(' ')
            text = ' '.join(parts[1:])
            if text:
                if update.message.from_user.username:
                    _from = '@' + update.message.from_user.username
                else:
                    _from = update.message.from_user.full_name
                conversation += f'{_from}:\n{text}'

            conversation = conversation.strip()
            ai = AICompliment(
                config=self.config,
                openai=self.openai,
            )
            msg = await ai.get_compliment(chat_id=chat_id, conversation=conversation, sender=_sender)
            await self.safe_reply_to_message(
                update,
                msg,
            )

        await wrap_with_indicator(
            update=update,
            context=context,
            coroutine=_run(),
            chat_action=constants.ChatAction.TYPING,
        )

    async def ximage(self, update: Update, context) -> None:
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        self.config.set_last_chat_user_command(chat_id, user_id, 'ximage')
        await self.safe_reply_to_message(update, "Please send me an image and your question as a caption.")

    async def append(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return

        text = update.message.text
        parts = text.split(' ')
        command = parts[0]
        key = parts[1]
        value = ' '.join(parts[2:])

        assert command == '/append'
        self.config.append(key, value)
        reply = f'SET {key}={self.config[key]}'
        logging.info(reply)

        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def questions_to_file(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return
        file_content = self.config['RAW_QUESTIONS']

        filename = '/tmp/questions.txt'
        with open(filename, 'w') as file:
            file.write(file_content)

        await update.effective_message.reply_document(
            message_thread_id=get_thread_id(update),
            document=filename,
            caption='Questions DB',
        )

        os.remove(filename)

    async def get_config_to_file(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return
        lines = []
        all_keys = sorted(self.config['ALL_KEYS']) + ['RAW_QUESTIONS']
        for k in all_keys:
            v = self.config[k]
            if len(str(v)) < 80:
                lines.append(f'{str(k)}={str(v)}')
            else:
                lines.append(f'{str(k)}=\n{str(v)}')
                lines.append('====')

        msg = '\n'.join(lines)
        await self.safe_send_file(
            file_content=msg,
            filename='config.txt',
            caption='Config',
            update=update,
        )

    async def safe_send_file(
            self,
            file_content: str,
            filename,
            caption,
            update: Update,
    ) -> None:
        await self.safe_telegram_request(
            update.effective_message.reply_document,
            message_thread_id=get_thread_id(update),
            document=file_content.encode('utf-8'),
            filename=filename,
            caption=caption,
        )
        self.message_queue.enqueue_message(Message(
            chat_id=str(update.effective_chat.id),
            content=f'File sent: {filename}',
            channel='telegram',
            user_id='ai',
            username='ai',
        ))

    async def get_config(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self.log_command(update.effective_chat.id, update)

        user_id = update.message.from_user.id
        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not allowed to use this command.")
            return
        lines = ['*Config\\:*']
        all_keys = sorted(self.config['ALL_KEYS'])
        for k in all_keys:
            v = self.config[k]
            if len(str(v)) < 80:
                lines.append(f'{str(k)}={str(v)}')
            else:
                lines.append(f'{str(k)}=\n{str(v)}')
                lines.append('=' * 10)

        msg = '\n'.join(lines)
        chunks = split_into_chunks(msg)
        for index, transcript_chunk in enumerate(chunks):
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=transcript_chunk,
                # parse_mode=constants.ParseMode.MARKDOWN_V2,
            )

    async def start(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        self.log_command(update.effective_chat.id, update)

        chat_id = update.effective_chat.id
        user_id = update.message.from_user

        _user_id = update.message.from_user.id if update.message and update.message.from_user else None
        username = (await self.get_username_by_user_id(_user_id)) if _user_id else None

        self.config.r.lpush('openai_kwargs_queue', json.dumps({
            "reply": '/start',
            "chat_id": chat_id,
            'openai_kwargs': {'messages': []},
            "telegram_username": username,
        }, ensure_ascii=False, indent=4))

        start_text = self.config['START_TEXT']
        await update.message.reply_text(start_text, disable_web_page_preview=True)

    async def all_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.log_command(update.effective_chat.id, update)

        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            f'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        user2period2tokens = {}
        user2period2cost = {}

        all_usage_trackers = get_all_usage_trackers()
        for user_id, usage in all_usage_trackers.items():
            tokens_today, tokens_month = all_usage_trackers[user_id].get_current_token_usage()
            tokens_all_time = all_usage_trackers[user_id].get_all_time_chat_tokens()
            user2period2tokens[user_id] = {}
            user2period2tokens[user_id]['today'] = tokens_today
            user2period2tokens[user_id]['month'] = tokens_month
            user2period2tokens[user_id]['all-time'] = tokens_all_time

            current_cost = all_usage_trackers[user_id].get_current_cost()
            user2period2cost[user_id] = {}
            user2period2cost[user_id]['today'] = round(current_cost['cost_today'], 4)
            user2period2cost[user_id]['month'] = round(current_cost['cost_month'], 4)
            user2period2cost[user_id]['all-time'] = round(current_cost['cost_all_time'], 4)

        for period in ['all-time', 'month', 'today']:
            user2tokens = {user_id: period2tokens[period] for user_id, period2tokens in user2period2tokens.items() if
                           period2tokens[period] > 0}
            user2cost = {user_id: period2cost[period] for user_id, period2cost in user2period2cost.items() if
                         period2cost[period] > 0}

            usage_text = f"""
===={period.upper()}====
Total users: {len(user2tokens)}
Total tokens processed: {sum(user2tokens.values())}
Total USD spent: {sum(user2cost.values())}
""".strip()

            for user_id, tokens in sorted(user2tokens.items(), key=lambda kv: -kv[1]):
                # # username = await self.get_username_by_user_id(user_id)
                # try:
                #     username, first_name, last_name = await self.get_user_username_firstname_lastname(user_id, chat_id=)
                # except Exception as e:
                #     logging.exception(f'exception on get_user_username_firstname_lastname for {user_id=}')
                #     raise
                user_name = all_usage_trackers[user_id].user_name
                # usage_text += f'\n@{username} "{first_name} {last_name}" used {tokens} tokens, cost ${user2cost[user_id]}'
                usage_text += f'\n{user_name} used {tokens} tokens, cost ${user2cost[user_id]}'

            chunks = split_into_chunks(usage_text)
            for index, transcript_chunk in enumerate(chunks):
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=transcript_chunk,
                )

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        self.log_command(update.effective_chat.id, update)

        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            f'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                     f'requested their usage statistics')

        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (transcribe_minutes_today, transcribe_seconds_today, transcribe_minutes_month,
         transcribe_seconds_month) = self.usage[user_id].get_current_transcription_duration()
        current_cost = self.usage[user_id].get_current_cost()

        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = await self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, update)
        bot_language = self.config['BOT_LANGUAGE']
        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            f"----------------------------\n"
        )
        text_today = (
            f"*{localized_text('usage_today', bot_language)}:*\n"
            f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
            f"{images_today} {localized_text('stats_images', bot_language)}\n"
            f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.4f}\n"
            f"----------------------------\n"
        )
        text_month = (
            f"*{localized_text('usage_month', bot_language)}:*\n"
            f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
            f"{images_month} {localized_text('stats_images', bot_language)}\n"
            f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.4f}"
        )
        # text_budget filled with conditional content
        text_budget = "\n\n"
        budget_period = self.config['BUDGET_PERIOD']
        # if remaining_budget < float('inf'):
        text_budget += (
            f"{localized_text('stats_budget', bot_language)}"
            f"{localized_text(budget_period, bot_language)}: "
            f"${remaining_budget:.4f}.\n"
        )
        # add OpenAI account information for admin request
        if is_admin(self.config, user_id):
            text_budget += (
                f"{localized_text('stats_openai', bot_language)}"
                f"{self.openai.get_billing_current_month():.4f}"
            )

        usage_text = text_current_conversation + text_today + text_month + text_budget
        # await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)
        await update.message.reply_text(usage_text)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        self.log_command(update.effective_chat.id, update)
        self.message_queue.enqueue_message(Message(
            channel='telegram',
            user_id=str(update.message.from_user.id),
            username=update.message.from_user.username,
            chat_id=str(update.effective_chat.id),
            content=update.message.text,
        ))

        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            f'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)

        if is_true(os.environ.get('RE')):
            self.field_storage.reset(chat_id, fields=self.bot_logic.ALL_FIELDS)

        await self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        await self.safe_send_message_to_thread(update, localized_text('reset_done', self.config['BOT_LANGUAGE']))

    async def reset_scope(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        self.log_command(update.effective_chat.id, update)
        self.message_queue.enqueue_message(Message(
            channel='telegram',
            user_id=str(update.message.from_user.id),
            username=update.message.from_user.username,
            chat_id=str(update.effective_chat.id),
            content=update.message.text,
        ))

        if not await is_allowed(self.config, update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                            f'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        await self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        self.config.delete(f'scope_id:{chat_id}')
        await self.safe_send_message_to_thread(update, localized_text('reset_done', self.config['BOT_LANGUAGE']))

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using DALL·E APIs
        """
        self.log_command(update.effective_chat.id, update)
        if await self.check_if_paused(update):
            return

        if not is_true(self.config['ENABLE_IMAGE_GENERATION']):
            await self.safe_send_message_to_thread(
                update=update,
                text='Image generation is globally disabled.',
            )

        if not await self.check_allowed_and_within_budget(update, context):
            await self.safe_send_message_to_thread(
                update=update,
                text='Image generation is skipped because budget allowance is exceeded.',
            )

        key = f'chat:{update.effective_chat.id}:noimage'
        if self.config[key] and is_true(self.config[key]):
            await self.safe_send_message_to_thread(
                update=update,
                text='Image generation is disabled for this chat.',
            )
            return

        image_query = message_text(update.message)
        if image_query == '':
            _status, _msg = self.image_gen_limiter.is_allowed_for_user_group(
                group_id=update.effective_chat.id,
                user_id=update.message.from_user.id,
                get=True,
            )
            if not _status and not is_admin(self.config, update.message.from_user.id):
                await self.safe_send_message_to_thread(
                    update=update,
                    text=_msg,
                )
                return
            self.config.set_last_chat_user_command(
                chat_id=update.effective_chat.id,
                user_id=update.message.from_user.id,
                command='image',
            )
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text="Send the image description is the next message, e.g. 'cat'"
            )
            return

        await self._image(update, context, image_query)

    async def _image(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            image_query: str
    ):
        logging.info(f'New image generation request received from user {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')

        async def _generate():
            _status, _msg = self.image_gen_limiter.is_allowed_for_user_group(
                group_id=update.effective_chat.id,
                user_id=update.message.from_user.id,
            )
            if not _status and not is_admin(self.config, update.message.from_user.id):
                await self.safe_send_message_to_thread(
                    update=update,
                    text=_msg,
                )
                return

            try:
                image_url, image_size = await self.openai.generate_image(
                    prompt=image_query,
                    client=os.environ.get('CLIENT'),
                    channel='telegram',
                    chat_id=update.effective_chat.id,
                )
                await update.effective_message.reply_photo(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    photo=image_url
                )
                # add image request to users usage tracker
                # user_id = update.message.from_user.id
                # self.usage[user_id].add_image_request(image_size, self.config['IMAGE_PRICES'])
                # # add guest chat request to guest usage tracker
                # if str(user_id) not in self.config['ALLOWED_TELEGRAM_USER_IDS'].split(',') and 'guests' in self.usage:
                #     self.usage["guests"].add_image_request(image_size, self.config['IMAGE_PRICES'])

            except Exception as e:
                logging.exception(e)
                if 'Your request was rejected as a result of our safety system.' in str(e):
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        text='Your request was rejected as a result of our safety system.',
                    )
                elif 'TimedOut' not in str(type(e)):
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        reply_to_message_id=get_reply_to_message_id(self.config, update),
                        text=f"{localized_text('image_fail', self.config['BOT_LANGUAGE'])}: {str(type(e))}",
                        # parse_mode=constants.ParseMode.MARKDOWN
                    )

        await wrap_with_indicator(update, context, _generate(), constants.ChatAction.UPLOAD_PHOTO)

    async def set_file(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        self.log_command(update.effective_chat.id, update)

        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
                # parse_mode=constants.ParseMode.MARKDOWN
            )

        async def warn_msg_and_help(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"Incorrect command format ({msg}), "
                     "should be "
                     "\"/set key\" + attached text file. "
                     "Or use \"/set KEY VALUE\" or \"/set KEY=VALUE\" with no file attached.",
                # parse_mode=constants.ParseMode.MARKDOWN
            )

        user_id = update.message.from_user.id
        file_id = update.message.effective_attachment.file_id \
            if update.message.effective_attachment else None
        file_size = update.message.effective_attachment.file_size \
            if update.message.effective_attachment else None
        text = update.effective_message.caption
        file_name = update.message.effective_attachment.file_name

        if not is_admin(self.config, user_id):
            await warn_msg("You are not allowed to use this command.")
            return

        if file_size > 10 * 1024 * 1024:
            await warn_msg(f"File size ({file_size} bytes) is too big. Max file size is 10MB.")
            return

        if not file_name.endswith('.txt'):
            await warn_msg(f"File name should end with .txt, but it is {file_name}")
            return

        parts = text.split(' ')
        if len(parts) != 2:
            await warn_msg_and_help(f"len(parts) != 2, {len(parts)=}")
            return
        if parts[0] != '/set':
            await warn_msg_and_help(f"parts[0] != '/set', {parts[0]=}")
            return
        key = parts[1]
        if '=' in key:
            await warn_msg_and_help(f"key contains '=', {key=}")
            return

        import io
        buf = io.BytesIO()
        file = await self.safe_telegram_request(self.application.bot.get_file, file_id)
        await file.download_to_memory(out=buf)
        file_content = buf.getvalue().decode('utf-8')
        logging.info(f'downloaded {len(file_content)=}')

        value = file_content
        reply = f'SET {key}={value}'
        logging.info(reply)
        self.config[key] = value

        await self.safe_send_message_to_thread(update, reply, technical=True)

    async def just_set_questions(self, content):
        questions = content.split('\n\n\n')
        questions = [_.strip() for _ in questions if _.strip()]
        logging.info(f'load {len(questions)=}')
        logging.info(f'{questions[0]=}')
        logging.info(f'{questions[-1]=}')
        await self.openai.semantic_db.recreate_index(questions, filename='db.txt')

    async def add_website(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            raw_markdown=False,
    ):
        if is_true(self.config.get_setup_group_disabled()):
            await self.safe_reply_to_message(update, "This command is disabled.")
            return

        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
            )

        parts = update.message.text.split(' ')
        website = parts[1]
        rest = ' '.join(parts[2:]).strip()

        user_id = update.message.from_user.id
        _user_id = update.message.from_user.id if update.message and update.message.from_user else None
        username = (await self.get_username_by_user_id(_user_id)) if _user_id else None
        WEBSITES_ADDER = False
        if username:
            WEBSITES_ADDER_USERNAMES = self.config['WEBSITES_ADDER_USERNAMES']
            if WEBSITES_ADDER_USERNAMES:
                for _ in str(WEBSITES_ADDER_USERNAMES).split(','):
                    if username.lstrip('@').lower() == _.lstrip('@').lower() or str(_user_id) == str(_):
                        WEBSITES_ADDER = True
                        break
        IS_ADMIN = is_admin(self.config, user_id)
        if not IS_ADMIN and not WEBSITES_ADDER:
            logging.warning(f'User {username=} {user_id=} is not allowed to use this command.')
            await warn_msg(f"You are not allowed to use this command. Your username {username=} is not admin.")

        number_of_user_demos = self.config[f'cnt_demo:{_user_id}'] or 0
        scope_id = self.get_next_scope_id()
        name = rest

        if not name and not raw_markdown:
            await warn_msg(
                f'You should specify NAME of the website, like:\n/add_website https://google.com/ Google LLC')
            return

        if self.config[f'scope:{scope_id}:ADMIN'] and self.config[f'scope:{scope_id}:ADMIN'] != _user_id:
            message = f'You are not the owner of {scope_id}, this CODE is already used by someone else.'
            await warn_msg(message)
            return
        else:
            self.config[f'scope:{scope_id}:ADMIN'] = _user_id
            self.config[f'scope:{scope_id}:NAME'] = name if name else website
            self.config[f'scope:{scope_id}:WEBSITE'] = website

        number_of_user_demos += 1
        if number_of_user_demos > 10_000 and not IS_ADMIN:
            await warn_msg("You created too many demo.")
            return
        else:
            self.config.set(f'cnt_demo:{_user_id}', number_of_user_demos)

        self.log_command(update.effective_chat.id, update)  # note caption is used

        await self.safe_send_message_to_thread(update, f'START {website=}, wait ~1min')
        await self._add_website(
            update,
            website,
            scope_id,
            verbose=True,
            raw_markdown=raw_markdown,
        )
        if not raw_markdown:
            link = f'https://eros-ai.cloud/chat.html?scopeId={scope_id}&port={os.environ.get("INSTAGRAM_HTTPS_PORT")}'
            await self.safe_send_message_to_thread(
                update,
                f"{name} WebChat - {link}",
            )

    async def add_website_to_main_scope(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            recreate=False,
    ):
        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
            )

        parts = update.message.text.split(' ')
        website = parts[1]

        user_id = update.message.from_user.id
        _user_id = update.message.from_user.id if update.message and update.message.from_user else None
        username = (await self.get_username_by_user_id(_user_id)) if _user_id else None
        IS_ADMIN = is_admin(self.config, user_id)
        if not IS_ADMIN:
            logging.warning(f'User {username=} {user_id=} is not allowed to use this command.')
            await warn_msg(f"You are not allowed to use this command. Your username {username=} is not admin.")

        scope_id = None

        self.log_command(update.effective_chat.id, update)  # note caption is used
        await self.safe_send_message_to_thread(update, f'START  {website=}, wait ~1min')
        await self._add_website(
            update, website, scope_id, verbose=True, recreate=recreate,
        )

    async def _add_website(
            self,
            update: t.Optional[Update],
            website: t.Optional[str],
            scope_id: t.Optional[str],
            verbose: bool = True,
            force_load_to_db: bool = False,
            hidden_code=False,
            recreate=True,
            raw_markdown=False,
    ):
        logging.info(f'add_website {website=} {scope_id=}')

        from tools.get_website_content import crawl_website

        report = []
        if not hidden_code:
            report.append(f'CODE=`{scope_id}`')
        report.append(f'{website=}')
        try:
            result = await crawl_website(
                website,
                max_urls=int(self.config['ADD_WEBSITE_MAX_URLS'] or 10),
                cache=False,
            )

            if raw_markdown:
                all_markdown = '\n\n'.join(f'# file:{k}\n========\n{v}\n========' for k, v in result.items())
                await self.safe_send_file(
                    update=update,
                    file_content=all_markdown,
                    caption=website,
                    filename='raw_markdown.md',
                )
                return

            no_load_to_DB = is_true(self.config['ADD_WEBSITE_NO_LOAD_TO_DB']) and not force_load_to_db

            from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
            import tiktoken
            encoding = tiktoken.encoding_for_model('gpt-4')  # todo be flexible

            def tokens(text):
                num_tokens = len(encoding.encode(text))
                return num_tokens

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,
                chunk_overlap=20,
                length_function=tokens
            )

            total_pages = 0
            failed_pages = 0
            for i_source, (source, content) in enumerate(result.items()):
                total_pages += 1

                split_start_at = time.time()
                docs = text_splitter.create_documents([content])

                max_docs = 300
                if len(docs) > max_docs:
                    logging.warning(f'add_website: too many chunks {source=} {len(docs)=}, cut to {max_docs}')
                    docs = docs[:max_docs]

                chunks = [_.page_content for _ in docs]
                chunks_sizes = [self.openai.count_tokens_of_string(_) for _ in chunks]
                logging.info(f'add_website: first, split to chunks {source=} {len(docs)=}, '
                             f'total chunks tokens {sum(chunks_sizes)}, max chunk size: {max(chunks_sizes)}, time: {time.time() - split_start_at:.2f} sec')

                try:
                    if no_load_to_DB:
                        if verbose:
                            report.append(f'{source=} OK ({len(chunks)})')
                        continue
                    kb = self.knowledge_base_manager.add_document(  # todo out of RAM soon
                        scope_id=scope_id,
                        filename=source,
                        text=content,
                    )
                    start_at = time.time()
                    if i_source == 0 and recreate:
                        await self.openai.semantic_db.recreate_index(
                            chunks,
                            filename=source,
                            scope_id=scope_id,
                            metadata_defaults={
                                'knowledge_base_id': kb.id,
                            }
                        )
                    else:
                        await self.openai.semantic_db.load_questions(
                            chunks,
                            filename=source,
                            scope_id=scope_id,
                            metadata_defaults={
                                'knowledge_base_id': kb.id,
                            }
                        )
                    end_at = time.time()
                    logging.info(f'add_website: loaded {source=} {len(chunks)=} in {end_at - start_at:.2f} sec')
                except Exception as exc:
                    failed_pages += 1
                    logging.exception(f'failed to process website load_questions {source=} {type(exc)=} {exc=}')
                    if verbose:
                        report.append(f'{source=} ERROR {type(exc)=}')
                else:
                    if verbose:
                        report.append(f'{source=} OK ({len(chunks)})')
            report.append(f'{total_pages=}, {failed_pages=}')
        except Exception as exc:
            logging.exception(f'failed to process website {website=} {type(exc)=} {exc=}')

            for _index, _chunk in enumerate(
                    split_into_chunks(f'failed to process website {website=} {type(exc)=} {exc=}')):
                if update:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=_chunk,
                    )
            success = 'FAIL'
            total_pages = 0
            failed_pages = 0
            message = str(exc)[:100]
        else:
            logging.info(f'add_website: OK {website=} {scope_id=} {total_pages=} {failed_pages=}')
            result = '\n'.join(report)
            for _index, _chunk in enumerate(split_into_chunks(result)):
                if update:
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=_chunk,
                    )
            if total_pages == 0:
                success = 'HMM'
                message = 'no pages processed'
            elif failed_pages / total_pages > 0.4:
                success = 'HMM'
                message = 'too many pages failed'
            else:
                success = 'SUCCESS'
                message = 'OK'
        return success, total_pages, failed_pages, message

    async def add_websites(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        self.log_command(update.effective_chat.id, update)  # note caption is used

        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
            )

        async def info_msg(msg: str):
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'{msg}',
            )

        user_id = update.message.from_user.id
        file_id = update.message.effective_attachment.file_id \
            if update.message.effective_attachment else None
        file_size = update.message.effective_attachment.file_size \
            if update.message.effective_attachment else None
        text = update.effective_message.caption
        file_name = update.message.effective_attachment.file_name

        _user_id = update.message.from_user.id if update.message and update.message.from_user else None
        username = (await self.get_username_by_user_id(_user_id)) if _user_id else None
        WEBSITES_ADDER = False
        if username:
            WEBSITES_ADDER_USERNAMES = self.config['WEBSITES_ADDER_USERNAMES']
            if WEBSITES_ADDER_USERNAMES:
                for _ in str(WEBSITES_ADDER_USERNAMES).split(','):
                    if username.lstrip('@').lower() == _.lstrip('@').lower() or str(_user_id) == str(_):
                        WEBSITES_ADDER = True
                        break
        IS_ADMIN = is_admin(self.config, user_id)
        if not IS_ADMIN and not WEBSITES_ADDER:
            logging.warning(f'User {username=} {user_id=} is not allowed to use this command.')
            await warn_msg(f"You are not allowed to use this command. Your username {username=} is not admin.")

        cnt_demo = self.config[f'cnt_demo:{_user_id}'] or 0

        if file_size > 50 * 1024 * 1024:
            await warn_msg(f"File size ({file_size} bytes) is too big. Max file size is 50MB.")
            return

        import io
        buf = io.BytesIO()
        file = await self.application.bot.get_file(file_id)
        await file.download_to_memory(out=buf)
        file_content = buf.getvalue()

        fpath = f'/tmp/{file_name}'
        with open(fpath, 'wb') as fout:
            fout.write(file_content)

        logging.info(f'downloaded {len(file_content)=} of {file_name=}')

        try:
            import csv
            queue = asyncio.Queue()

            results = []

            async def worker():
                """Worker to process tasks from the queue."""
                while not queue.empty():
                    sz = get_directory_size()
                    if sz > 10 * 10 ** 9:
                        logging.warning(f'get_directory_size()={sz} > 10 * 10**9, break')
                        break

                    try:
                        _row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await _process_row(_row)
                    queue.task_done()

            async def _process_row(row):
                """Process a single row."""
                try:
                    NAME = row["NAME"].strip()
                    WEBSITE = row["WEBSITE"].strip()
                    CODE = row["CODE"].strip()

                    if not NAME or not WEBSITE or not CODE:
                        logging.info(f'add_websites {NAME=} {WEBSITE=} {CODE=}')
                        await warn_msg(f'SKIP {NAME=} {WEBSITE=} {CODE=}')
                        return

                    if self.config[f'scope:{CODE}:message']:
                        logging.info(f'_process_row ALREADY {NAME=} {WEBSITE=} {CODE=}')
                        success = self.config[f'scope:{CODE}:success']
                        total_pages = self.config[f'scope:{CODE}:total_pages']
                        failed_pages = self.config[f'scope:{CODE}:failed_pages']
                        message = self.config[f'scope:{CODE}:message']
                        r = {
                            'NAME': NAME,
                            'CODE': CODE,
                            'WEBSITE': WEBSITE,
                            'success': success,
                            'total_pages': total_pages,
                            'failed_pages': failed_pages,
                            'message': message,
                        }
                        for k, v in row.items():
                            if k not in r:
                                r[k] = v
                        results.append(r)
                        return
                    logging.info(f'_process_row START {NAME=} {WEBSITE=} {CODE=}')

                    if self.config[f'scope:{CODE}:ADMIN'] and self.config[f'scope:{CODE}:ADMIN'] != user_id:
                        success = 'NOT_OWNER'
                        total_pages = 0
                        failed_pages = 0
                        message = f'You are not the owner of {CODE}, this CODE is already used by someone else.'
                    else:
                        self.config[f'scope:{CODE}:NAME'] = NAME
                        self.config[f'scope:{CODE}:WEBSITE'] = WEBSITE
                        self.config[f'scope:{CODE}:ADMIN'] = user_id

                        success, total_pages, failed_pages, message = await self._add_website(
                            update=update,
                            website=WEBSITE,
                            scope_id=CODE,
                            verbose=False,
                        )
                        if is_true(self.config['ADD_WEBSITE_NO_LOAD_TO_DB']):
                            message = f'lazy: {message}'

                    r = {
                        'NAME': NAME,
                        'CODE': CODE,
                        'WEBSITE': WEBSITE,
                        'success': success,
                        'total_pages': total_pages,
                        'failed_pages': failed_pages,
                        'message': message,
                    }
                    for k, v in row.items():
                        if k not in r:
                            r[k] = v
                    results.append(r)
                    self.config[f'scope:{CODE}:success'] = success
                    self.config[f'scope:{CODE}:total_pages'] = total_pages
                    self.config[f'scope:{CODE}:failed_pages'] = failed_pages
                    self.config[f'scope:{CODE}:message'] = message

                except Exception as exc:
                    logging.exception(f'failed to process row {type(exc)=} {exc=}')
                    await warn_msg(f'failed to process row {type(exc)=} {exc=}, {row=}')

            with open(fpath, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                # lines = list(reader)[:500]
                lines = list(reader)
                cnt_demo += len(lines)
                if cnt_demo > 10_000 and not IS_ADMIN:
                    await warn_msg("You created too many demo.")
                    return
                else:
                    self.config.set(f'cnt_demo:{_user_id}', cnt_demo)

                for i_row, row in enumerate(lines):
                    await queue.put(row)

            N_WORKERS = 10
            workers = [worker() for _ in range(N_WORKERS)]
            await asyncio.gather(*workers)
        except Exception as exc:
            logging.exception(f'failed to process file {type(exc)=} {exc=}')

            for _index, _chunk in enumerate(split_into_chunks(f'ERROR to process websites {type(exc)=} {exc=}')):
                await self.safe_telegram_request(
                    update.effective_message.reply_text,
                    message_thread_id=get_thread_id(update),
                    text=_chunk,
                )
        else:

            try:
                result_csv_path = '/tmp/websites.csv'
                with open(result_csv_path, 'w', newline='') as csvfile:
                    fieldnames = list(results[0].keys())
                    for r in results:
                        for k in r:
                            if k not in fieldnames:
                                fieldnames.append(k)
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in results:
                        writer.writerow(row)

                await self.safe_telegram_request(
                    update.effective_message.reply_document,
                    message_thread_id=get_thread_id(update),
                    document=result_csv_path,
                    caption='websites',
                )
                os.remove(result_csv_path)
            except Exception as exc:
                for _index, _chunk in enumerate(split_into_chunks(f'failed to reply with CSV {type(exc)=} {exc=}')):
                    await self.safe_telegram_request(
                        update.effective_message.reply_text,
                        message_thread_id=get_thread_id(update),
                        text=_chunk,
                    )
        finally:
            os.remove(fpath)

    async def add_file(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        from tools.get_text_from_file import extract_from_file

        self.log_command(update.effective_chat.id, update)  # note caption is used

        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
            )

        async def warn_msg_and_help(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"I can process .txt .doc .docx .pdf .xls .xlsx",
            )

        user_id = update.message.from_user.id
        file_id = update.message.effective_attachment.file_id \
            if update.message.effective_attachment else None
        file_size = update.message.effective_attachment.file_size \
            if update.message.effective_attachment else None
        text = update.effective_message.caption
        file_name = update.message.effective_attachment.file_name

        parts = text.split(' ')
        rest = ' '.join(parts[1:]).strip()
        if rest:
            scope_id = rest
        else:
            scope_id = None

        if not is_admin(self.config, user_id):
            await warn_msg("You are not allowed to use this command.")
            return

        if file_size > 50 * 1024 * 1024:
            await warn_msg(f"File size ({file_size} bytes) is too big. Max file size is 50MB.")
            return

        import io
        buf = io.BytesIO()
        file = await self.safe_telegram_request(self.application.bot.get_file, file_id)
        await file.download_to_memory(out=buf)
        file_content = buf.getvalue()

        fpath = f'/tmp/{file_name}'
        with open(fpath, 'wb') as fout:
            fout.write(file_content)

        logging.info(f'downloaded {len(file_content)=} of {file_name=}')

        try:
            from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
            import tiktoken
            encoding = tiktoken.encoding_for_model('gpt-4')  # todo be flexible

            def tokens(text):
                num_tokens = len(encoding.encode(text))
                return num_tokens

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=30,
                length_function=tokens
            )

            result = extract_from_file(fpath)
            for source, content in result.items():
                docs = text_splitter.create_documents([content])
                logging.info(f'split to chunks {source=} {len(docs)=}')

                chunks = [_.page_content for _ in docs]

                logging.info(f'add_file {source=}')
                logging.info(f'add_file {content=}')
                logging.info(f'add_file {chunks=}')

                await self.openai.semantic_db.load_questions(chunks, filename=source, scope_id=scope_id)
                await self.safe_reply_to_message(update, f'loaded {source=} {len(chunks)=} {scope_id=}')
        except Exception as exc:
            logging.exception(f'failed to process file {type(exc)=} {exc=}')
            await self.safe_reply_to_message(update, f'failed to process file {type(exc)=} {exc=}')
        finally:
            os.remove(fpath)

    def get_next_scope_id(self):
        next_scope_id = int(self.config.r.incr('next_scope_id'))
        next_scope_id = 7700000 + next_scope_id
        return str(next_scope_id)

    async def add_partner(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
            )

        parts = update.message.text.split(' ')
        username = parts[1]

        user_id = update.message.from_user.id

        if not is_admin(self.config, user_id):
            await warn_msg(f"You {user_id=} are not allowed to use this command.")
            return

        WEBSITES_ADDER_USERNAMES = self.config['WEBSITES_ADDER_USERNAMES']
        if WEBSITES_ADDER_USERNAMES:
            WEBSITES_ADDER_USERNAMES = str(WEBSITES_ADDER_USERNAMES).split(',')
        else:
            WEBSITES_ADDER_USERNAMES = []
        if username not in WEBSITES_ADDER_USERNAMES:
            WEBSITES_ADDER_USERNAMES.append(username)
        self.config['WEBSITES_ADDER_USERNAMES'] = ','.join(WEBSITES_ADDER_USERNAMES)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=f'ok, {username=} added, {WEBSITES_ADDER_USERNAMES=}',
        )

    async def nedviga(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        try:
            from tools.get_text_from_file import extract_from_file

            self.log_command(update.effective_chat.id, update)  # note caption is used

            user_id = update.message.from_user.id
            file_id = update.message.effective_attachment.file_id \
                if update.message.effective_attachment else None
            file_size = update.message.effective_attachment.file_size \
                if update.message.effective_attachment else None
            file_name = update.message.effective_attachment.file_name \
                if update.message.effective_attachment else None
            text = update.effective_message.caption if file_id else update.effective_message.text
            if text.startswith('/nedviga'):
                text = text[len('/nedviga'):]

            chat_id = update.effective_chat.id
            user_id = update.message.from_user

            _user_id = update.message.from_user.id if update.message and update.message.from_user else None
            username = (await self.get_username_by_user_id(_user_id)) if _user_id else None

            logging.info(f'nedviga {chat_id=}, {username=}, {file_id=}, {file_size=}, {file_name=}')
            logging.info(f'nedviga {text=}')

            file_sources = {}
            if file_id:
                import io
                buf = io.BytesIO()
                file = await self.application.bot.get_file(file_id)
                await file.download_to_memory(out=buf)
                file_content = buf.getvalue()

                fpath = f'/tmp/{file_name}'
                with open(fpath, 'wb') as fout:
                    fout.write(file_content)

                logging.info(f'downloaded {len(file_content)=} of {file_name=}')

                try:
                    file_sources = extract_from_file(fpath)
                except Exception as exc:
                    logging.exception(f'failed to process file {type(exc)=} {exc=}')
                finally:
                    os.remove(fpath)
            logging.info(f'nedviga {list(file_sources)=}')

            mega_result = []

            if not file_sources:
                user_msg = f"""
----
user message:
{text}
----
""".strip()
                texts = []
                if text.strip():
                    texts.append(user_msg)

                conversation = [
                    {'role': 'system', 'content': self.config['JUST_PROMPT_2']},
                    {'role': 'user', 'content': '\n'.join(texts)},
                ]
                logging.info(f'nedviga conversation: {pprint.pformat(conversation)}')

                response, openai_kwargs = await self.openai.just_get_chat_response(
                    conversation=conversation,
                    chat_id='nedviga',
                    with_kwargs=True,
                )
                self.config.r.lpush('openai_kwargs_queue', json.dumps({
                    "reply": response,
                    "chat_id": chat_id,
                    'openai_kwargs': openai_kwargs,
                    "telegram_username": username,
                }, ensure_ascii=False, indent=4))

                sources = []
                if text.strip():
                    sources.append('chat message')
                sources_str = "\n".join(' - ' + _ for _ in sources)
                chat_response = f"""
processed sources:
{sources_str}

result:
{response}
"""
                logging.info(f'nedviga result:\n{chat_response}')
                mega_result.append(chat_response.split('====')[-1].strip())

            for source, content in file_sources.items():
                user_msg = f"""
----
user message:
{text}
----
""".strip()
                content_msg = f"""
----
source {source}:
{content}
----
""".strip()
                texts = []
                if text.strip():
                    logging.info(f'use user message')
                    texts.append(user_msg)
                if content.strip():
                    logging.info(f'use content message')
                    texts.append(content_msg)
                else:
                    logging.info(f'no content message')
                    continue

                conversation = [
                    {'role': 'system', 'content': self.config['JUST_PROMPT_2']},
                    {'role': 'user', 'content': '\n'.join(texts)},
                ]
                logging.info(f'nedviga conversation: {pprint.pformat(conversation)}')

                response, openai_kwargs = await self.openai.just_get_chat_response(
                    conversation=conversation,
                    chat_id='nedviga',
                    with_kwargs=True,
                )
                self.config.r.lpush('openai_kwargs_queue', json.dumps({
                    "reply": response,
                    "chat_id": chat_id,
                    'openai_kwargs': openai_kwargs,
                    "telegram_username": username,
                }, ensure_ascii=False, indent=4))

                sources = []
                if text.strip():
                    sources.append('chat message text')
                sources.append(source)

                sources_str = "\n".join(' - ' + _ for _ in sources)
                chat_response = f"""
processed sources:
{sources_str}

result:
{response}
"""
                logging.info(f'nedviga result:\n{chat_response}')
                mega_result.append(chat_response.split('====')[-1].strip())

            mega_result = '\n\n'.join(mega_result)
            for _index, _chunk in enumerate(split_into_chunks(mega_result)):
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=_chunk,
                )
        except Exception as exc:
            logging.exception(f'failed to process nedviga {type(exc)=} {exc=}')
            for _index, _chunk in enumerate(split_into_chunks(f'failed to process file {type(exc)=} {exc=}')):
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=_chunk,
                )

    async def sql(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        self.log_command(update.effective_chat.id, update)  # note caption is used

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        text = update.effective_message.text
        if text.startswith('/sql'):
            text = text[len('/sql'):]
        text = text.strip()

        if not text:
            await self.safe_reply_to_message(update, f'Empty question! The format is /sql QUESTION')
            return

        if not is_admin(self.config, user_id):
            await self.safe_reply_to_message(update, "You are not admin!")
            return

        logging.info(f'sql: {text=}')

        typing_task = asyncio.create_task(self.action_typing_forever(update))
        analysist = SQLQuestioner(
            config=self.config,
        )
        try:
            reply = await analysist.answer_sql_question(text)
        except Exception as exc:
            logging.exception(f'failed to process etherscan {type(exc)=} {exc=}')
            reply = f'I cannot give you a clear answer 🙂\nWe are actively improving AI Agent ⚙️\nTry to ask something else 😉'
        finally:
            typing_task.cancel()
        await self.safe_reply_to_message(update, reply)

    async def etherscan(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        self.log_command(update.effective_chat.id, update)  # note caption is used

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        text = update.effective_message.text
        if text.startswith('/etherscan'):
            text = ' '.join(text.split(' ')[1:])
        text = text.strip()

        if not text:
            self.config.set_last_chat_user_command(chat_id, user_id, 'etherscan:address')
            await self.safe_reply_to_message(
                update,
                f'What is the address of the smart contract you want to analyze?',
            )
            return

        address = text.split(' ')[0]
        question = ' '.join(text.split(' ')[1:])
        await self._etherscan(update, context, address, question)

    async def _etherscan(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            address: str,
            question: str = '',
    ):
        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id

        if not is_admin(self.config, user_id) and not self.limiter.check_limits(user_id=user_id, chat_id=chat_id,
                                                                                multiplier=20):
            await self.safe_reply_to_message(update, "You used your daily limit of messages.")
            return

        logging.info(f'etherscan: {address=}')
        logging.info(f'etherscan: {question=}')

        typing_task = asyncio.create_task(self.action_typing_forever(update))
        if not question:
            analysist = SmartContractAnalysist(
                config=self.config,
                openai=self.openai,
            )
            try:
                reply = await analysist.process_address(address)
            except Exception as exc:
                logging.exception(f'failed to process etherscan {type(exc)=} {exc=}')
                reply = f'failed to process etherscan {type(exc)=}'
            finally:
                typing_task.cancel()
            await self.safe_reply_to_message(update, reply)
        else:
            from ai_tools.contract_questioner import ContractQuestioner

            async def get_answer(model):
                try:
                    questioner = ContractQuestioner(self.config, model=model)
                    reply = await questioner.answer_contract_question(contract=address, question=question)
                except Exception as exc:
                    logging.exception(f'failed to process ContractQuestioner {type(exc)=} {exc=}')
                    reply = f'I cannot give you a clear answer 🙂\nWe are actively improving AI Web3 Agent ⚙️\nTry to ask something else 😉'
                else:
                    if reply.startswith('{'):
                        logging.warning(f'bad {reply=}')
                        reply = f'I cannot give you a clear answer 🙂\nWe are actively improving AI Web3 Agent ⚙️\nTry to ask something else 😉'
                return reply

            try:
                reply = await get_answer(model='gpt-3.5-turbo-16k')  # 3.5
                if reply.startswith('I cannot give you a clear answer'):
                    logging.warning(f'failed with gpt-3.5 try again with gpt-3.5')
                    reply = await get_answer(model='gpt-3.5-turbo-16k')  # 3.5
                    if reply.startswith('I cannot give you a clear answer'):
                        logging.warning(f'failed with gpt-3.5 try again with gpt-4')
                        self.limiter.check_limits(user_id=user_id, chat_id=chat_id, multiplier=50)
                        reply = await get_answer(model='gpt-4')  # 4
            except Exception as exc:
                logging.exception(f'failed to process etherscan {type(exc)=} {exc=}')
                reply = f'I cannot give you a clear answer 🙂\nWe are actively improving AI Web3 Agent ⚙️\nTry to ask something else 😉'
            finally:
                typing_task.cancel()
            await self.safe_reply_to_message(update, reply)

    async def mem(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        self.log_command(update.effective_chat.id, update)  # note caption is used

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        username = update.message.from_user.username
        text = update.effective_message.text

        if update.message.chat.type == 'private':
            if not is_admin(self.config, user_id, username=username) and not \
                    is_mems_admin(self.config, user_id, username=username):
                await self.safe_reply_to_message(update,
                                                 "You are not admin.")
                return
            scope_id = self.get_update_scope_id(update)
        else:
            is_group_admin = await self.check_if_group_admin(chat_id=chat_id, user_id=user_id)
            if not is_admin(self.config, user_id, username=username) and not is_group_admin:
                await self.safe_reply_to_message(update,
                                                 "You are not admin.")
                return
            scope_id = self.get_update_scope_id(update)

        if text and text.startswith('/mem'):
            text = text[len('/mem'):]
        text = text.strip()

        if not text:
            await self.safe_reply_to_message(update, f'Empty!')
            return

        await self._mem(scope_id, text)

        response = await self.openai.semantic_db.get_mems(scope_id=scope_id)
        total = len(response.documents)
        await self.safe_reply_to_message(update, f'Remembered!\nThere are {total} facts in the Knowledge base now!')

    def get_update_scope_id(self, update) -> int | str | None:
        chat_id = update.effective_chat.id
        if update.message.chat.type == 'private':
            scope_id = self.config.get_chat_scope_id('private')
        else:
            scope_id = self.config.get_chat_scope_id(chat_id)
        return scope_id

    async def _mem(
        self,
        scope_id: str | int | None,
        text: str,
        metadata_defaults: dict = None,
    ):
        kb = self.knowledge_base_manager.add_document(
            scope_id=scope_id,
            filename='mem',
            text=text,
        )

        questions = [
            text
        ]
        await self.openai.semantic_db.load_questions(
            questions,
            filename='mem',
            scope_id=scope_id,
            metadata_defaults={
                **(metadata_defaults or {}),
                'knowledge_base_id': kb.id,
            }
        )
        m = {
                **(metadata_defaults or {}),
                'knowledge_base_id': kb.id,
            }
        logging.info(f'zzzzzzzzzzzzzzzzzz {m=}')

        logging.info(f'scope_id: {scope_id}, mem: {text=}')

    async def set_questions_file(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            mem=False,
    ):
        self.log_command(update.effective_chat.id, update)  # note caption is used

        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
                # parse_mode=constants.ParseMode.MARKDOWN
            )

        async def warn_msg_and_help(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f"Incorrect command format ({msg}), "
                     "should be "
                     "\"/set_questions\" + attached text file. "
                     "File should have texts parts separated with 2 blank newlines.\n"
                     "\n"
                     "\n"
                     "<b>Example.txt:</b> <code>\n"
                     "Question: What is the answer to life, the universe and everything?\n"
                     "Answer: 42\n"
                     "\n"
                     "\n"
                     "Question: What is China?\n"
                     "Answer: China is an asian country\n"
                     "\n"
                     "\n"
                     "Question: What is emergency phone number?\n"
                     "Answer: 911 </code>",
                parse_mode=constants.ParseMode.HTML
            )

        user_id = update.message.from_user.id
        file_id = update.message.effective_attachment.file_id \
            if update.message.effective_attachment else None
        file_size = update.message.effective_attachment.file_size \
            if update.message.effective_attachment else None
        text = update.effective_message.caption
        file_name = update.message.effective_attachment.file_name

        if not is_admin(self.config, user_id):
            await warn_msg("You are not allowed to use this command.")
            return

        if file_size > 10 * 1024 * 1024:
            await warn_msg(f"File size ({file_size} bytes) is too big. Max file size is 10MB.")
            return

        if not file_name.endswith('.txt'):
            await warn_msg(f"File name should end with .txt, but it is {file_name}")
            return

        parts = text.split(' ')
        # if len(parts) != 1:
        #     await warn_msg_and_help(f"len(parts) != 1, {len(parts)=}")
        #     return
        if parts[0] not in ['/set_questions', '/sq', '/mem']:
            await warn_msg_and_help(f"parts[0] != '/set_questions', {parts[0]=}")
            return

        rest = ' '.join(parts[1:]).strip()
        if rest:
            scope_id = rest
        else:
            scope_id = None

        import io
        buf = io.BytesIO()
        file = await self.safe_telegram_request(self.application.bot.get_file, file_id)
        await file.download_to_memory(out=buf)
        file_content = buf.getvalue().decode('utf-8')
        logging.info(f'downloaded {len(file_content)=}')

        if '====' in file_content:
            questions = file_content.split('====')
        elif '\r\n\r\n\r\n' in file_content:  # windows style
            questions = file_content.split('\r\n\r\n\r\n')
        else:
            questions = file_content.split('\n\n\n')

        questions = [_.strip() for _ in questions if _.strip()]
        logging.info(f'load {len(questions)=}')
        logging.info(f'{questions[0]=}')
        logging.info(f'{questions[-1]=}')

        try:
            reply = f"""Successfully load questions file.
Use {scope_id=}
Starting to recreate index.
{len(questions)=} questions loaded.

⏳⏳⏳ WAIT FOR A FEW MINUTES ⏳⏳⏳
"""
            chunks = split_into_chunks(reply)
            for index, transcript_chunk in enumerate(chunks):
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=transcript_chunk,
                    # parse_mode=constants.ParseMode.HTML
                )

            start_at = time.time()
            if mem:
                logging.info(f'load mems')
                await self.openai.semantic_db.load_questions(questions, filename=file_name, scope_id=scope_id)
            else:
                logging.info(f'recreate_index')
                await self.openai.semantic_db.recreate_index(questions, filename=file_name, scope_id=scope_id)
            logging.info(f'recreate_index took {time.time() - start_at} seconds')

            reply = f"""🎉🎉🎉 Successfully processed questions file {scope_id=}! 🎉🎉🎉
Took {round(time.time() - start_at, 2)} seconds to recreate index.
{len(questions)=} questions loaded.
"""

            self.config['RAW_QUESTIONS'] = file_content

            chunks = split_into_chunks(reply)
            for index, transcript_chunk in enumerate(chunks):
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=transcript_chunk,
                    # parse_mode=constants.ParseMode.HTML
                )
        except Exception as exc:
            logging.exception(exc)
            # todo fail safe
            await warn_msg(f"Failed to set questions file contact ADMIN to check VectorDB! {exc=}")

    async def receive_text_document(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
    ):
        self.log_command(update.effective_chat.id, update)  # note use caption

        async def warn_msg(msg: str):
            logging.warning(msg)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=f'WARNING: {msg}',
                # parse_mode=constants.ParseMode.MARKDOWN
            )

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        file_id = update.message.effective_attachment.file_id \
            if update.message.effective_attachment else None
        file_unique_id = update.message.effective_attachment.file_unique_id \
            if update.message.effective_attachment else None
        file_size = update.message.effective_attachment.file_size \
            if update.message.effective_attachment else None
        if not file_id:
            await warn_msg('No file_id found for receive_text_document')
            return

        text = update.effective_message.caption
        file_name = update.message.effective_attachment.file_name
        logging.info(f'{file_name=} {file_id=} {file_unique_id=} {file_size=} from {chat_id=} with {text=}')

        command = text.split(' ')[0]
        if command == '/set':
            await self.set_file(update, context)
        elif command in ['/set_questions', '/sq']:
            await self.set_questions_file(update, context)
        elif command in ['/mem']:
            await self.set_questions_file(update, context, mem=True)
        elif command in ['/set_from_file', '/sff']:
            await self.set_from_file(update, context)
        elif command == '/add_file':
            await self.add_file(update, context)
        elif command == '/add_websites':
            await self.add_websites(update, context)
        elif command == '/nedviga':
            await self.nedviga(update, context)
        else:
            logging.info(f'ignore file processing')
            if not is_group_chat(update):
                if is_admin(self.config, user_id):
                    await warn_msg(
                        'files processing is not supported (except of /set /set_questions /set_from_file, /af commands)')
                else:
                    await warn_msg('files processing is not supported')
            return

    async def send_to_debug_chats(self, message: str):
        DEBUG_CHATS = self.config['DEBUG_CHATS']
        if not DEBUG_CHATS:
            return
        chat_ids = [int(_) for _ in DEBUG_CHATS.split(',')]
        await self.send_message_to_chat_ids(chat_ids, message)

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        self.log_command(update.effective_chat.id, update)
        if await self.check_if_paused(update):
            return

        if not self.config['ENABLE_TRANSCRIPTION'] or not await self.check_allowed_and_within_budget(update, context):
            return

        if is_group_chat(update) and is_true(self.config['IGNORE_GROUP_TRANSCRIPTIONS']):
            logging.info(f'Transcription coming from group chat, ignoring...')
            return

        chat_id = update.effective_chat.id

        logging.info(f'transcribe: {update.message.effective_attachment=}')

        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f'{filename}.mp3'
            bot_language = self.config['BOT_LANGUAGE']
            try:
                media_file = await context.bot.get_file(update.message.effective_attachment.file_id)
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=(
                        f"{localized_text('media_download_fail', bot_language)[0]}: "
                        f"{str(e)}. {localized_text('media_download_fail', bot_language)[1]}"
                    ),
                    # parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            try:
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(f'New transcribe request received from user {update.message.from_user.name} '
                             f'(id: {update.message.from_user.id})')

            except Exception as e:
                logging.exception(e)
                # await update.effective_message.reply_text(
                #     message_thread_id=get_thread_id(update),
                #     reply_to_message_id=get_reply_to_message_id(self.config, update),
                #     text=localized_text('media_type_fail', bot_language)
                # )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            try:
                transcript = await self.openai.transcribe(
                    filename_mp3,
                    chat_id=chat_id,
                    user_id=user_id,
                    channel='telegram',
                )

                transcription_price = self.config['TRANSCRIPTION_PRICE']
                self.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                allowed_user_ids = self.config['ALLOWED_TELEGRAM_USER_IDS'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                # check if transcript starts with any of the prefixes
                response_to_transcription = any(transcript.lower().startswith(prefix.lower()) if prefix else False
                                                for prefix in self.config['VOICE_REPLY_PROMPTS'])

                if self.config['VOICE_REPLY_WITH_TRANSCRIPT_ONLY'] and not response_to_transcription:

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(self.config, update) if index == 0 else None,
                            text=transcript_chunk,
                            # parse_mode=constants.ParseMode.MARKDOWN
                        )
                else:
                    # Get the response of the transcript
                    key = f'replying_{chat_id}_{user_id}'
                    try:
                        await self._prompt(
                            update=update,
                            context=context,
                            replying_key=key,
                            prompt=transcript,
                        )
                    finally:
                        self.config.delete(key)

            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    # parse_mode=constants.ParseMode.MARKDOWN
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(update, context, _execute(), constants.ChatAction.TYPING)

    async def safe_telegram_request(self, callable, *args, max_retries=5, **kwargs):
        delay = 5
        i_try = 0
        while True:
            i_try += 1
            try:
                return await callable(*args, **kwargs)
            except telegram.error.TimedOut as exc:
                logging.exception(f'TimedOut {i_try=}, {type(exc)=}, {exc=}')
                if i_try >= max_retries:
                    raise
                await asyncio.sleep(delay)
            except telegram.error.RetryAfter as exc:
                logging.exception(f'RetryAfter {i_try=}, {type(exc)=}, {exc=}')
                if i_try >= max_retries:
                    raise
                await asyncio.sleep(exc.retry_after)
            except telegram.error.BadRequest as exc:
                logging.exception(f'BadRequest {i_try=}, {type(exc)=}, {exc=}')
                if i_try >= max_retries:
                    raise
                await asyncio.sleep(1)
            except telegram.error.NetworkError as exc:
                logging.exception(f'NetworkError {i_try=}, {type(exc)=}, {exc=}')
                if i_try >= max_retries:
                    raise
                await asyncio.sleep(delay)
            except telegram.error.Forbidden:
                raise

    async def get_username_by_user_id(self, user_id):
        try:
            chat = await self.safe_telegram_request(self.application.bot.get_chat, chat_id=user_id)
        except telegram.error.BadRequest as exc:
            if 'Chat not found' in str(exc):
                return f'UnknownChat for {user_id=}'
            else:
                raise
        username = chat.username
        if username:
            return username
        else:
            return chat.first_name or user_id

    async def get_user_username_firstname_lastname(self, user_id, chat_id=None):
        try:
            chat = await self.safe_telegram_request(self.application.bot.get_chat, chat_id=user_id)
        except telegram.error.BadRequest:
            if chat_id is None:
                raise
            chat_member = await self.safe_telegram_request(self.application.bot.get_chat_member, chat_id=chat_id,
                                                           user_id=user_id)
            chat = chat_member.user
        return chat.username, chat.first_name, chat.last_name

    def remove_replying_keys(self):  # todo be careful about instagram
        for key in self.config.get_keys_by_pattern('replying_*'):
            self.config.delete(key)

    async def send_no_website_message(self, update, scope_id, chat_id):
        text = f"""
        You have entered an invalid code.

        💡You can find your activation code in the message you received from us.
        🧑🏽‍💻 Support email — vadim@salesbotai.co
        """.strip()
        for _index, _chunk in enumerate(split_into_chunks(text)):
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text=_chunk,
            )

        _user_id = update.message.from_user.id if update.message and update.message.from_user else None
        username = (await self.get_username_by_user_id(_user_id)) if _user_id else None

        self.config.r.lpush('openai_kwargs_queue', json.dumps({
            "reply": scope_id,
            "chat_id": chat_id,
            'openai_kwargs': {'messages': []},
            "username": username,
            "channel": 'telegram',
            'msg_id': 777,
        }))

    async def builder_flow(
            self,
            user_id,
            chat_id,
            update,
    ):
        if self.config[f'setup:{user_id}:step'] == 'verify' and user_id == chat_id:
            from w3tools import check_wallet

            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text

            status, message = check_wallet(
                signature=text,
                group_id=group_id,
                MIN_AIX_BALANCE=int(self.config['MIN_AIX_BALANCE'] or 100 * 1000 * 10 ** 18),
                whitelist=(self.config['WHITELIST'] or '').split(','),
            )

            if not status:
                await self.send_continue_setup(
                    update=update,
                    user_id=user_id,
                    group_id=group_id,
                    prefix=f'Verification failed with the message: {message}!\n'
                )
            else:
                self.config[f'group_settings:{group_id}:verified'] = text
                await self.send_continue_setup(
                    update=update,
                    user_id=user_id,
                    group_id=group_id,
                    prefix=f'Verified! {message}\n'
                )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'company_info' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            self.config[f'group_settings:{group_id}:company_info'] = text
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'Company Info: {text}!\n----\n\n'
            )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'dex_tools_link' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            self.config[f'group_settings:{group_id}:dex_tools_link'] = text
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'DEX Tools link: {text}!\n----\n\n'
            )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'token_info' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            self.config[f'group_settings:{group_id}:token_info'] = text
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'Token Info: {text}!\n----\n\n'
            )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'how_to_buy' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            self.config[f'group_settings:{group_id}:how_to_buy'] = text
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'How to buy: {text}!\n----\n\n'
            )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'contracts' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            self.config[f'group_settings:{group_id}:contracts'] = text
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'Contracts: {text}!\n----\n\n'
            )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'website' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text

            # if self.config[f'scope_id:{group_id}']:
            #     scope_id = self.config[f'scope_id:{group_id}']
            # else:
            #     scope_id = self.get_next_scope_id()
            #     self.config[f'scope_id:{group_id}'] = scope_id

            # todo resolve - update scope not recreate

            scope_id = self.get_next_scope_id()

            await self.safe_send_message_to_thread(update, f'Start processing website: {text}\nWait for 3 minutes!')

            try:
                websites = []
                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    websites.append(line)

                exists = await self.openai.semantic_db.scope_exists(scope_id=scope_id)
                logging.info(f'{exists=} {scope_id=}')
                if exists:
                    await self.openai.semantic_db.drop_index(scope_id=scope_id)
                await self.openai.semantic_db.create_index(scope_id=scope_id)

                for i_website, website in enumerate(websites):
                    logging.info(f'SETUP: {scope_id=}')
                    await self._add_website(
                        update=update,
                        website=website,
                        scope_id=scope_id,
                        force_load_to_db=True,
                        hidden_code=True,
                        recreate=False,
                    )
            except Exception as exc:
                logging.exception(f'failed to add website {type(exc)=} {exc=}')
                await self.send_continue_setup(
                    update=update,
                    user_id=user_id,
                    group_id=group_id,
                    prefix=f'Failed to process WebSite: "{text}"!\n----\n\n'
                )
            else:
                self.config[f'group_settings:{group_id}:website'] = text
                await self.send_continue_setup(
                    update=update,
                    user_id=user_id,
                    group_id=group_id,
                    prefix=f'WebSite: {text}!\n----\n\n'
                )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'ton_of_voice' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            self.config[f'group_settings:{group_id}:ton_of_voice'] = text
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'Ton Of Voice: {text}!\n----\n\n'
            )
            return True
        elif self.config[f'setup:{user_id}:step'] == 'coingecko' and user_id == chat_id:
            del self.config[f'setup:{user_id}:step']
            group_id = self.config[f'setup:{user_id}:group_id']
            text = update.effective_message.text
            text = text.rstrip('/')

            if text.startswith('https://www.coingecko.com/'):
                self.config[f'group_settings:{group_id}:coingecko'] = text
                await self.send_continue_setup(
                    update=update,
                    user_id=user_id,
                    group_id=group_id,
                    prefix=f'CoinGecko Link: {text}!\n----\n\n'
                )
            else:
                await self.send_continue_setup(
                    update=update,
                    user_id=user_id,
                    group_id=group_id,
                    prefix=f'Invalid CoinGecko, valid example: https://www.coingecko.com/en/coins/aigentx\n----\n\n'
                )
            return True
        elif self.config[f'setup:{user_id}:group_id'] and user_id == chat_id:
            group_id = self.config[f'setup:{user_id}:group_id']
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f'Unknown command. Press some button to continue!\n'
            )
            return True
        else:
            return False

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            chat_id = update.effective_chat.id

            if update.effective_chat.type == 'channel':
                logging.info(f'channel {chat_id=} {update.effective_chat=}')
                user_id = update.effective_chat.id
            else:
                user_id = update.effective_message.from_user.id
        except AttributeError:
            logging.info(f'{update.effective_message=}')
            logging.info(f'{update.message=}')
            logging.exception(f'prompt AttributeError')
            raise

        last_command = self.config.get_last_chat_user_command(chat_id, user_id)
        if last_command == 'etherscan:address':
            address = update.effective_message.text
            await self.safe_reply_to_message(
                update,
                f'The address is {address}. What is your question?',
            )
            self.config.set_last_chat_user_command(chat_id, user_id, 'etherscan:question')
            self.config.set_last_chat_user_etherscan_address(chat_id, user_id, address)
            return
        elif last_command == 'etherscan:question':
            await self._etherscan(
                update=update,
                context=context,
                address=self.config.get_last_chat_user_etherscan_address(chat_id, user_id),
                question=update.effective_message.text,
            )
            self.config.delete_last_chat_user_command(chat_id, user_id)
            self.config.delete_last_chat_user_etherscan_address(chat_id, user_id)
            return
        elif last_command == 'image':
            image_query = update.effective_message.text
            self.config.delete_last_chat_user_command(chat_id, user_id)
            await self._image(update, context, image_query)
            return

        builder_flow_triggered = await self.builder_flow(
            chat_id=chat_id, user_id=user_id, update=update)
        if builder_flow_triggered:
            return

        scope_id = self.get_update_scope_id(update)
        if self.config[f'group_settings:{chat_id}:website']:
            await self.openai.update_assistant_prompt(
                group_id=chat_id,
                channel='telegram',
                _dextools=self.dextools,
                _erc20prices=self.erc20prices,
            )

        logging.info(f'prompt {chat_id=} {user_id=} {scope_id=} {update.effective_message.text=}')

        if update.edited_message:
            logging.info(f'_prompt SKIP: update.edited_message, {update=}')
            return

        if not update.effective_message:
            logging.info(f'_prompt SKIP: not update.effective_message, {update=}')
            return

        if update.effective_message.via_bot:
            logging.info(f'_prompt SKIP: {update.effective_message.via_bot=}, {update=}')
            return

        # if not scope_id and is_true(
        #         self.config['STRICT_SCOPE_ID']):  # scope_id could be None if not STRICT_SCOPE_ID
        #     await self.no_scope_id_flow(chat_id)
        #     return

        key = f'replying_{chat_id}_{user_id}'
        prompt = message_text(update.effective_message)
        try:
            return await self._prompt(update, context, replying_key=key, prompt=prompt)
        finally:
            self.config.delete(key)

    async def realtime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._prompt(
            update=update,
            context=context,
            prompt=f'Give me realtime token info (price, market capitalization, how many holders, etc), formatted as a list with emoji. '
                   f'Send links to coingecko and dextools if you have.',
            explicit_call=True,
        )

    def get_user_username(self, user):
        if user.username:
            return user.username
        parts = [
            user.first_name or '',
            user.last_name or '',
            # f'#{user.id}'
        ]
        return ' '.join(_ for _ in parts if _.strip())

    def get_message_username(self, message: TelegramMessage):
        user = message.from_user
        return self.get_user_username(user)

    def check_group_ingored_allowed(self, chat_id):
        IGNORE_GROUP_IDS = (self.config['IGNORE_GROUP_IDS'] or '').strip()
        if IGNORE_GROUP_IDS and \
                (str(chat_id) in IGNORE_GROUP_IDS.split(',') or IGNORE_GROUP_IDS == '*'):
            logging.info(f'Group {chat_id} is ignored')
            return False

        ALLOWED_GROUP_IDS = (self.config['ALLOWED_GROUP_IDS'] or '').strip()
        if ALLOWED_GROUP_IDS and \
                (str(chat_id) not in ALLOWED_GROUP_IDS.split(',') or ALLOWED_GROUP_IDS == '*'):
            logging.info(f'Group {chat_id} is not allowed')
            # await self.safe_send_message_to_thread(update, 'I am not allowed to talk here :(')  # todo
            return False
        return True

    async def check_if_paused(self, update) -> bool:
        if is_true(self.config['PAUSE']):
            if self.config['PAUSE_REASON']:
                reason = self.config['PAUSE_REASON']
            else:
                reason = 'Paused!'
            await self.safe_send_message_to_thread(update, reason)
            return True
        return False

    async def _prompt(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            prompt: str,
            replying_key: str = None,
            explicit_call=False,
    ):
        """
        React to incoming messages and respond accordingly.

        Prompt may come from transcribe or from prompt command.
        """
        sent_message = None

        await self.log_incoming_message(update)

        if await self.check_if_paused(update):
            return

        chat_id = update.effective_chat.id
        if update.effective_chat.type == 'channel':
            _user_id = update.effective_chat.id
            username = f'channel-{update.effective_chat.id}'
            scope_id = self.config.get(f'scope_id:{chat_id}')
        else:
            if not update.message.from_user:
                logging.info(f'_prompt SKIP: not from_user')
                return
            scope_id = self.config.get_chat_scope_id(chat_id)
            _user_id = update.message.from_user.id
            username = self.get_user_username(update.message.from_user)
            logging.info(
                f'_prompt: New message {update.message.from_user=}, {chat_id=}, {scope_id=}, {username=}')

        if chat_id:
            chat_name = update.effective_chat.effective_name
            self.config.set(f'name:chat:{chat_id}', chat_name, log=False)
        if _user_id:
            self.config.set(f'name:user:{_user_id}', username, log=False)

        bot_username = self.application.bot.username

        if is_group_chat(update):
            if update.effective_message.reply_to_message and not self.last_messages_manager.get_message_by_ref(
                    update.effective_message.reply_to_message):
                self.last_messages_manager.save_message(update.effective_message.reply_to_message)  # save chain
            if update.effective_message and not self.last_messages_manager.get_message_by_ref(update.effective_message):
                self.last_messages_manager.save_message(update.effective_message)
                # save this because it will be responded by bot

        if is_true(self.config['EXCHANGER_TOUCH']):
            exchanger = ExchangerFlow(self.config, self)
            await exchanger.process(
                update=update,
                chat_id=chat_id,
                user_id=_user_id,
                text=prompt,
            )
            return
        elif is_true(os.environ.get('RE')):
            async def on_finish(bot_message, user_message):
                await self.safe_send_message_to_thread(update, bot_message)

            async def on_exception():
                await self.safe_send_message_to_thread(update, f'Technical problems, please try again later')

            await self.bot_wrapper.process_message(
                chat_id=chat_id,
                username=username,
                channel='telegram',
                message=prompt,
                on_finish=on_finish,
                on_exception=on_exception,
            )
            return

        is_x = False
        try:
            if update.effective_message.text and (
                    update.effective_message.text.startswith('/x ')
                    or
                    update.effective_message.text.startswith('/X ')
                    or
                    update.effective_message.text.startswith(f'/x@{bot_username}')
                    or
                    update.effective_message.text.startswith(f'/X@{bot_username}')
            ):
                is_x = True
        except Exception as exc:
            logging.exception(f'unknown {exc}')

        max_tokens = int(self.config['MAX_REPLY_TOKENS'])

        _ensure_has_answer = False  # should AI ensure it has a useful answer before sending anything
        if is_group_chat(update):
            if not self.check_group_ingored_allowed(chat_id):
                return

            if explicit_call:
                logging.info(f'explicit_call {explicit_call=}')
                allowing = True
            elif is_x:
                logging.info(f'{prompt=}')
                logging.info('Message is started with /x or /X, allowing...')
                allowing = True
            elif bot_username in prompt:
                logging.info(f'{prompt=}')
                logging.info('Message contains bot username, allowing...')
                allowing = True
            elif update.effective_message.reply_to_message and \
                    update.effective_message.reply_to_message.from_user.id == self.application.bot.id:
                logging.info(f'{prompt=}')
                logging.info('Message is a reply to the bot, allowing...')
                allowing = True
            elif str(chat_id) in (self.config['ALWAYS_REPLY_GROUP'] or ''.split(',')):
                logging.info('Message is in always reply group, allowing...')
                allowing = True
            elif self.config['GROUP_TRIGGER_KEYWORD'] and (username is None or not username.lower().endswith('bot')):
                allowing = False
                for keyword in self.config['GROUP_TRIGGER_KEYWORD'].split(','):
                    if keyword.lower() in prompt.lower():
                        allowing = True
                        logging.info(f'{prompt=}')
                        logging.info(f'Message is started with trigger "{keyword}", allowing...')
                        break
            else:
                allowing = False

            _flag = self.config[f'group_settings:{chat_id}:autoreply']
            if not allowing and _flag and is_true(_flag):
                _status, _explain = await self.is_relevant_group_message.detect(
                    chat_id=chat_id,
                    messages=[prompt],
                )
                logging.info(f'is_relevant_group_message {_status=}')
                logging.info(f'is_relevant_group_message {_explain=}')
                allowing = _status if isinstance(_status, bool) else False
                if allowing:
                    max_tokens = int(self.config['MAX_REPLY_TOKENS'])
                    max_tokens = max(20, int(max_tokens / 2))
                    _ensure_has_answer = True

            if allowing:
                previous_replies = self.last_messages_manager.fetch_previous_replies(update.message)

                if not prompt and is_x:
                    logging.info(f'prompt: is empty and is_x, so add some')
                    if previous_replies:
                        prompt = ''
                    else:
                        prompt = 'Hello!'

                new_prompts = []
                if previous_replies:
                    logging.info(f'Previous replies len: {len(previous_replies)}')
                    logging.info(f'Previous replies:\n{pprint.pformat(previous_replies)}')
                    for reply in reversed(previous_replies):
                        try:
                            if (reply.text and not reply.text.startswith(f'{self.get_message_username(reply)}: ')):
                                new_prompt = f'@{self.get_message_username(reply)}: {reply.text}\n\n'
                            else:
                                new_prompt = reply.text
                            if new_prompt:
                                new_prompts.append(new_prompt)
                            else:
                                logging.warning(f'Empty {new_prompt=}')
                        except Exception as exc:
                            logging.exception(f'unknown {exc}')
                new_prompt = f'@{self.get_message_username(update.message)}: {prompt}'
                new_prompts.append(new_prompt)
                prompt = new_prompts
                logging.info(f'Prompt is now group of messages:\n{prompt}')
            else:
                logging.info('Message does not start with trigger keyword, ignoring...')
                return

        else:
            if not is_true(self.config['PRIVATE_CHATTING_ENABLED']) \
                    and not is_admin(self.config, update.effective_message.from_user.id):
                logging.info('Private chatting is disabled, ignoring...')
                reply = self.config['PRIVATE_CHATTING_DISABLED_REPLY']
                if reply:
                    await self.safe_send_message_to_thread(update, reply)
                return

        if not is_admin(self.config, _user_id):
            limits_check = self.limiter.check_limits(_user_id, chat_id)
            if not limits_check:
                await self.safe_reply_to_message(update,
                                                 'You have reached your daily limit of messages. Please try again tomorrow.')
                return

        if not await self.check_allowed_and_within_budget(update, context):
            return

        if replying_key:
            if self.config[replying_key] == 'true':
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=self.config['ONE_MESSAGE_PER_TIME'],
                )
                return
            self.config.setex(replying_key, 'true', 180)

        typing_task = asyncio.create_task(self.action_typing_forever(update))
        try:
            await self.stream_reply(
                update=update,
                context=context,
                chat_id=chat_id,
                _user_id=_user_id,
                username=username,
                prompt=prompt,
                scope_id=scope_id,
                max_tokens=max_tokens,
                ensure_has_answer=_ensure_has_answer,
            )
        except Exception as exc:
            logging.exception(exc)
            await self.send_to_debug_chats(f'Exception while processing message, '
                                           f'user: {_user_id}, '
                                           f'chat_id: {chat_id}, '
                                           f'username: @{username}\n'
                                           f'{type(exc)=}\n'
                                           f'{exc=}\n'
                                           f'{traceback.format_exc()}')

            EXCEPTION_USER_MESSAGE = self.config['EXCEPTION_USER_MESSAGE'] or ''
            if is_true(self.config['SHOW_EXCEPTION_DETAILS_TO_USER']):
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{EXCEPTION_USER_MESSAGE}\n{localized_text('chat_fail', self.config['BOT_LANGUAGE'])} {str(exc)}",
                )
            else:
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    text=f"{EXCEPTION_USER_MESSAGE}",
                )
        finally:
            try:
                if is_true(self.config[f'SHOW_QUERIES:{chat_id}']):
                    await asyncio.sleep(0.05)
                    queries = self.config[f'SHOW_QUERIES:{chat_id}:last_queries']
                    await self.safe_send_message_to_thread(update, f'DEBUG Queries:\n{pprint.pformat(queries)}')
            finally:
                typing_task.cancel()

    def save_msg_if_need(self, update, context, _msg, reply_to=None):
        if not (
                is_group_chat(update)
                or int(self.config['MAX_HISTORY_SIZE']) == 0
        ):
            return
        if _msg:
            if not _msg.reply_to_message and reply_to:
                kwargs = json.loads(_msg.to_json())
                kwargs['reply_to_message'] = json.loads(reply_to.to_json())
                _msg = telegram.Message.de_json(kwargs, bot=context.bot)  # copy
            self.last_messages_manager.save_message(_msg)  # this is reply chain part (todo maybe not needed)
        else:
            pass

    async def action_typing_forever(
            self,
            update,
    ):
        while True:
            try:
                await self.safe_telegram_request(
                    update.effective_message.reply_chat_action,
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update)
                )
            except asyncio.CancelledError:
                logging.info(f'stop typing')
                break
            except Exception as exc:
                logging.exception(f'Exception while typing {type(exc)=} {exc=}')
                break
            else:
                await asyncio.sleep(5)

    async def handle_stream_response(
            self,
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            chat_id: int,
            stream_response,
            sent_message: TelegramMessage | None,  # if None does not update the message
    ):
        prev = ''
        backoff = 0
        stream_chunk = 0
        tokens_used_info = None
        content = ''
        i = 0
        openai_kwargs = None
        async for content, tokens, tokens_used_info, openai_kwargs in stream_response:
            if len(content.strip()) == 0:
                continue

            stream_chunks = split_into_chunks(content)
            if len(stream_chunks) > 1:
                logging.info(f'xxx xxx xxx does it ever go here?')
                content = stream_chunks[-1]
                if stream_chunk != len(stream_chunks) - 1:
                    stream_chunk += 1
                    try:
                        if sent_message:
                            _msg = await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                stream_chunks[-2])
                            self.save_msg_if_need(
                                update=update,
                                context=context,
                                _msg=_msg,
                                reply_to=update.effective_message
                            )
                    except Exception as exc:
                        logging.exception(f'Exception while editing/saving reply message, {type(exc)=}, {exc=}')
                    try:
                        if sent_message:
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                text=content if len(content) > 0 else "..."
                            )
                            logging.info(f'sent_message {content=} {sent_message.text=}')
                            self.save_msg_if_need(
                                update=update,
                                context=context,
                                _msg=sent_message,
                                reply_to=update.effective_message
                            )
                    except Exception as exc:
                        logging.exception(f'Exception while editing/saving reply message, {type(exc)=}, {exc=}')
                    continue

            cutoff = get_stream_cutoff_values(update, content)
            cutoff += backoff

            if i == 0:
                try:
                    if sent_message:
                        _msg = await edit_message_with_retry(
                            context,
                            chat_id,
                            str(sent_message.message_id),
                            text=content,
                        )
                        self.save_msg_if_need(
                            update=update,
                            context=context,
                            _msg=_msg,
                            reply_to=update.effective_message
                        )
                except Exception as exc:
                    logging.exception(f'Exception while editing/saving reply message, {type(exc)=} {exc=}')
                    continue

            elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                prev = content

                try:
                    if sent_message:
                        # use_markdown = tokens != 'not_finished'
                        use_markdown = True
                        _msg = await edit_message_with_retry(
                            context,
                            chat_id,
                            str(sent_message.message_id),
                            text=content,
                            markdown=use_markdown
                        )
                        self.save_msg_if_need(
                            update=update,
                            context=context,
                            _msg=_msg,
                            reply_to=update.effective_message
                        )
                except RetryAfter as e:
                    backoff += 5
                    await asyncio.sleep(e.retry_after)
                    continue

                except TimedOut:
                    backoff += 5
                    await asyncio.sleep(0.5)
                    continue

                except Exception as e:
                    logging.exception(f'Exception while editing/saving reply message {e}')
                    backoff += 5
                    continue

                await asyncio.sleep(0.001)

            i += 1

        return content, tokens_used_info, openai_kwargs

    async def post_validations(
            self,
            content,
            context,
            chat_id,
            sent_message,
            openai_kwargs,
            skip_hallucinations=False,
    ):
        validations = [
            {
                'condition': is_true(self.config['CUTOFF_INCOMPLETE_LAST_SENTENCE']),
                'log_message': 'final pretty cutoff',
                'validator': SentenceRefactor,
                'log_prefix': 'MSG CONTENT TRUNCATED BY SENTENCES'
            },
            {
                'condition': is_true(self.config['NO_HALLUCINATIONS']) and not skip_hallucinations,
                'log_message': 'no hallucinations',
                'validator': NoHallucinations(self.openai, openai_kwargs=openai_kwargs),
                'log_prefix': 'MSG NO HALLUCINATIONS'
            },
            {
                'condition': os.environ.get('CLIENT') == 'babytiger',
                'log_message': 'run links replacements',
                'validator': LinkReplacerBabyTiger(self.openai),
                'log_prefix': 'MSG LINKS REPLACER'
            },
            {
                'condition': is_true(self.config['USE_CALL_MANAGER']),
                'validator': CallManager(self.openai),
                'log_prefix': 'MSG CallManager'
            }
        ]

        for validation in validations:
            if not validation['condition']:
                continue
            logging.info(validation.get('log_message', ''))

            try:
                _status, new_content = await validation['validator'].validate(chat_id, content)
                if new_content != content:
                    logging.info(f'{validation["log_prefix"]}: \n'
                                 f'"""\n{content}\n""" --> """\n{new_content}\n"""')
                    content = new_content
                    if sent_message:
                        await edit_message_with_retry(
                            context=context,
                            chat_id=chat_id,
                            message_id=str(sent_message.message_id),
                            text=content,
                        )
            except Exception as exc:
                logging.exception(f'Exception while validation {type(exc)=} {exc=}')

        if content and os.environ.get('CLIENT') == 'moon':
            content += '\n\n' + '[Add AI to my Group](http://t.me/AIgentXBot) | by [AigentX](http://aigentx.xyz/)'
            if sent_message:
                await edit_message_with_retry(
                    context=context,
                    chat_id=chat_id,
                    message_id=str(sent_message.message_id),
                    text=content,
                    disable_web_page_preview=True,
                )

        return content

    async def stream_reply(
            self,
            update: Update,
            context,
            chat_id,
            _user_id,
            username,
            prompt: typing.Union[str, list[str]],  # for groups, it is list of last messages
            scope_id,
            max_tokens=None,
            ensure_has_answer=False,
    ):
        """
        Stream the response to the user.
        @param update: Telegram update
        @param context: Telegram context
        @param chat_id: Telegram chat id
        @param _user_id: Telegram user id
        @param username: Telegram username
        @param prompt: Prompt to send to the API (for groups, it is list of last messages)
        @param scope_id: Scope id
        @param max_tokens: Max tokens to use
        @param ensure_has_answer: Ensure that the response has an answer
        """
        stream_response = self.openai.get_chat_response_stream(
            chat_id=chat_id,  # not user_id
            query=prompt,
            scope_id=scope_id,
            max_tokens=max_tokens,
        )

        if not ensure_has_answer:
            sent_message = await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                reply_to_message_id=get_reply_to_message_id(self.config, update),
                text='...'
            )
        else:
            sent_message = None

        try:
            content, tokens_used_info, openai_kwargs = await self.handle_stream_response(
                update=update,
                context=context,
                chat_id=chat_id,
                stream_response=stream_response,
                sent_message=sent_message,
            )
        except Exception as exc:
            logging.exception(f'unknown exception in the middle of stream processing {type(exc)=}. {exc=}')
            _msg = await edit_message_with_retry(
                context,
                chat_id,
                str(sent_message.message_id),
                text="🛠️",  # maintenance
            )
            raise
        else:
            content = await self.post_validations(
                content=content,
                context=context,
                chat_id=chat_id,
                sent_message=sent_message,
                openai_kwargs=openai_kwargs,
            )

            if ensure_has_answer:
                is_valid_answer_detector = IsValidAnswer(openai=self.openai)
                is_valid_answer, _ = await is_valid_answer_detector.detect(chat_id, [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                ])
                if not is_valid_answer:
                    logging.info(f'User: {prompt=}')
                    logging.info(f'Assistant: {content=}')
                    logging.info(f'Not valid answer, skip')
                    return
                _msg = await self.safe_reply_to_message(
                    update=update,
                    text=content,
                )
                self.save_msg_if_need(
                    update=update,
                    context=context,
                    _msg=_msg,
                    reply_to=update.effective_message,
                )

        logging.info(f'response content {tokens_used_info=}: {content}')
        self.log_outcoming_message(chat_id, content)

        # todo several api calls
        self.config.r.lpush('openai_kwargs_queue', json.dumps({
            "reply": content,
            "chat_id": chat_id,
            'openai_kwargs': openai_kwargs,
            "telegram_username": username,
        }, ensure_ascii=False, indent=4))

        trigger_content = f'''\nUSER:\n{prompt}\n\nBOT:\n{content}'''
        MANAGER_TRIGGER = (self.config['MANAGER_TRIGGER'] if self.config['MANAGER_TRIGGER'] else '').strip()
        manager_triggered = False
        if not MANAGER_TRIGGER:
            pass
        else:
            for trigger in MANAGER_TRIGGER.split(','):
                trigger = trigger.lower()
                if trigger in trigger_content:
                    logging.info(f'MANAGER_TRIGGER {trigger=} {trigger_content=}')
                    manager_triggered = True
                    break

        if manager_triggered:
            task = {
                'chat_id': chat_id,
                'content': trigger_content,
                'channel': 'telegram',
                'username': username,
                'link': '',
            }
            logging.info(f'Adding triggered task {task=}')
            self.config.r.lpush('triggers', json.dumps(task))

        self.message_queue.enqueue_message(Message(
            chat_id=str(chat_id),
            content=content,
            user_id='ai',
            username='ai',
            channel='telegram',
            metadata={
                "manager_triggered": manager_triggered,
            }
        ))

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        self.log_command(update.effective_chat.id, update)
        logging.info(f'inline_query: {update.inline_query.query}')

        if await self.check_if_paused(update):
            return

        query = update.inline_query.query
        if len(query) < 3:
            return
        if not await self.check_allowed_and_within_budget(update, context, is_inline=True):
            return

        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f'{callback_data_suffix}{result_id}'

        await self.send_inline_query_result(update, result_id, message_content=query, callback_data=callback_data)

    async def send_inline_query_result(self, update: Update, result_id, message_content, callback_data=""):
        """
        Send inline query result
        """
        try:
            reply_markup = None
            bot_language = self.config['BOT_LANGUAGE']
            if callback_data:
                reply_markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton(text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                         callback_data=callback_data)
                ]])

            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content,
                thumb_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea'
                          '-b02a7a32149a.png',
                reply_markup=reply_markup
            )

            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.exception(f'An error occurred while generating the result card for inline query {e}')

    async def send_continue_setup(self, update, user_id, group_id, prefix):
        reply_markup = self.gen_reply_markup(group_id)
        group_name = await self.get_group_name(group_id)
        self.config[f'group_settings:{group_id}:group_name'] = group_name

#         if self.check_everything(group_id):
#             del self.config[f'setup:{user_id}:group_id']
#             del self.config[f'setup:{user_id}:step']
#             # scope_id = self.get_next_scope_id()
#             scope_id = self.config[f'scope_id:{group_id}']  # set in website
#             self.config[f'scope:{scope_id}:group_id'] = group_id
#             logging.info(f'New scope_id={scope_id} for group_id={group_id}')
#             await self.openai.update_assistant_prompt(
#                 group_id,
#                 channel='telegram',
#                 _dextools=self.dextools,
#                 _erc20prices=self.erc20prices,
#             )
#             # questions = ['.']
#             # await self.openai.semantic_db.recreate_index(questions, filename='db.txt', scope_id=scope_id)
#             port = os.environ.get("INSTAGRAM_HTTPS_PORT")
#             code = fr'''
# \<script src="https://eros-ai.cloud/aixchat.js"
#     data-project-id="{group_id}"
#     data-scope-id="{scope_id}"
#     data-api-url="https://eros-ai.cloud:{port}"
#     data-user-token="35e26211fa1d4746bc814f9cb2a478b8"
#     data-div-id="aibot"
# \>\<\/script\>
# '''.strip()
#             logging.info(f'webchat code:\n{code}')
#             await self.safe_send_message_to_thread(update, f"To use WebChat add this to your website inside \\<body\\>\n\n```\n{code}\n```", use_markdown=True)
#
#             link = f'https://eros-ai.cloud/chat.html?scopeId={scope_id}&port={port}'
#             await self.safe_send_message_to_thread(
#                 update,
#                 f"WebChat - {link}",
#             )
#
#             await self.safe_send_message_to_thread(update, f"🎉 Everything is set up for '{group_name}' #{group_id}!")
#         else:
        msg = f"{prefix}Continue to setup '{group_name}' #{group_id}"
        await self.safe_send_message_to_thread(update, msg, reply_markup=reply_markup)

    def check_everything(self, group_id):
        for _ in self.openai.GROUP_SETTINGS_KEYS:
            if not _['optional'] and not self.config[f'group_settings:{group_id}:{_["name"]}']:
                return False, _["name"]
        return True, None

    async def send_media_group(
            self,
            update: Update,
            media: list,
            caption: str = None,
    ):
        await self.safe_telegram_request(
            self.application.bot.send_media_group,
            chat_id=update.effective_chat.id,
            media=media,
            caption=caption,
        )

    async def handle_callback_inline_query(self, update: Update, context: CallbackContext):
        """
        Handle the callback query from the inline query result
        """
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        chat_id = update.callback_query.message.chat_id
        username = update.callback_query.from_user.username

        await update.callback_query.answer()

        if callback_data.startswith("cancel-pin:"):
            user_id = update.callback_query.from_user.id

            q_chat_id = callback_data.split(':')[1]
            q_user_id = callback_data.split(':')[2]

            if str(q_user_id) != str(user_id):
                await self.safe_send_message_to_thread(update, f'Wrong user')
                return

            is_group_admin = await self.check_if_group_admin(chat_id=chat_id, user_id=user_id)
            if not is_admin(self.config, user_id, username=username) and not is_group_admin:
                await self.safe_send_message_to_thread(update, f'Only admin can do this')
                return

            self.config.delete_last_chat_user_command(q_chat_id, q_user_id)
            await self.safe_send_message_to_thread(update, f'Pin canceled')
            return

        if callback_data.startswith("mems:"):
            user_id = update.callback_query.from_user.id
            is_group_admin = await self.check_if_group_admin(chat_id=chat_id, user_id=user_id)
            _is_mems_admin = is_mems_admin(self.config, user_id, username=username)
            if not is_admin(self.config, user_id, username=username) and not is_group_admin and not _is_mems_admin:
                await self.safe_send_message_to_thread(update, f'Only admin can do this')
                return
            scope_id = self.config.get_chat_scope_id(chat_id, no_default=True)  # could be None
            carousel = MemsCarousel(
                scope_id=scope_id,
                semantic_db_client=self.openai.semantic_db,
                send_message_to_chat_id=self.send_message_to_chat_id
            )
            await carousel.click(update, context)
            return

        if callback_data.startswith("pins:"):
            user_id = update.callback_query.from_user.id
            is_group_admin = await self.check_if_group_admin(chat_id=chat_id, user_id=user_id)
            _is_pins_admin = False
            if not is_admin(self.config, user_id, username=username) and not is_group_admin and not _is_pins_admin:
                await self.safe_send_message_to_thread(update, f'Only admin can do this')
                return
            carousel = PinsCarousel(
                config=self.config,
                chat_id=chat_id,
                semantic_db_client=self.openai.semantic_db,
                send_message_to_chat_id=self.send_message_to_chat_id,
            )
            await carousel.click(update, context)
            return

        if callback_data.startswith("ton_of_voice:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return
            ton_of_voice = callback_data.split(":")[2]
            self.config[f'group_settings:{group_id}:ton_of_voice'] = ton_of_voice
            del self.config[f'setup:{user_id}:step']
            await self.send_continue_setup(update, user_id, group_id, prefix=f'OK, {ton_of_voice=}\n')

        elif callback_data.startswith("verified:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            await self.send_media_group(
                update,
                [
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign1.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign2.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign3.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign4.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign5.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign6.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign7.png'), 'rb')),
                    InputMediaPhoto(open(os.path.join(STATIC_FOLDER, 'sign8.png'), 'rb')),
                ],
            )
            link = f'https://shy-wave-3009.on.fleek.co/?groupId={group_id}'
            await self.safe_send_message_to_thread(update,
                                                   f'🛡️ Approve AIX ownership via this link: {link}, and paste the signature here.')
            self.config[f'setup:{user_id}:step'] = 'verify'

        elif callback_data.startswith("company_info:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:company_info']
                await self.safe_send_message_to_thread(update, f'Company info:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            with open(os.path.join(STATIC_FOLDER, f'company_info.png'), "rb") as photo:
                await self.application.bot.send_photo(
                    chat_id=update.callback_query.message.chat_id,
                    photo=photo,
                )

            await self.safe_send_message_to_thread(update, f'Send your company info in 2-3 paragraphs:')
            self.config[f'setup:{user_id}:step'] = 'company_info'
        elif callback_data.startswith("website:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:website']
                await self.safe_send_message_to_thread(update, f'Websites:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            with open(os.path.join(STATIC_FOLDER, f'website.jpg'), "rb") as photo:
                await self.application.bot.send_photo(
                    chat_id=update.callback_query.message.chat_id,
                    photo=photo,
                )

            with open(os.path.join(STATIC_FOLDER, f'websites.png'), "rb") as photo:
                await self.application.bot.send_photo(
                    chat_id=update.callback_query.message.chat_id,
                    photo=photo,
                )

            await self.safe_send_message_to_thread(
                update,
                f'Send full links to your WebSites (e.g. https://aigentx.xyz/):\nNote, if you have multiple websites (e.g. whitepaper on GitBook), send them all line by line in ONE message.'
            )
            self.config[f'setup:{user_id}:step'] = 'website'
        elif callback_data.startswith("select_ton_of_voice:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(
                    update,
                    f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:ton_of_voice']
                await self.safe_send_message_to_thread(update, f'Style:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            reply_markup = self.gen_reply_markup_ton_of_voice(group_id)
            await self.safe_send_message_to_thread(update, f'Select your Ton of Voice:', reply_markup=reply_markup)
        elif callback_data.startswith("coingecko:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(
                    update,
                    f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:coingecko']
                await self.safe_send_message_to_thread(update, f'CoinGecko:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            with open(os.path.join(STATIC_FOLDER, f'coingecko.png'), "rb") as photo:
                await self.application.bot.send_photo(
                    chat_id=update.callback_query.message.chat_id,
                    photo=photo,
                )
            await self.safe_send_message_to_thread(update,
                                                   "Send your full CoinGecko link (e.g. https://www.coingecko.com/en/coins/aigentx):")
            self.config[f'setup:{user_id}:step'] = 'coingecko'
        elif callback_data.startswith("cancel:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            del self.config[f'setup:{user_id}:group_id']
            del self.config[f'setup:{user_id}:step']
            await self.safe_send_message_to_thread(update, f"Canceled")

        elif callback_data.startswith("finish:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            is_finished, unfinished_field_name = self.check_everything(group_id)
            if is_finished:
                del self.config[f'setup:{user_id}:group_id']
                del self.config[f'setup:{user_id}:step']
                scope_id = self.config.get_chat_scope_id(chat_id, no_default=True)  # set in website
                self.config[f'scope:{scope_id}:group_id'] = group_id
                logging.info(f'New scope_id={scope_id} for group_id={group_id}')
                await self.openai.update_assistant_prompt(
                    group_id,
                    channel='telegram',
                    _dextools=self.dextools,
                    _erc20prices=self.erc20prices,
                )
                port = os.environ.get("INSTAGRAM_HTTPS_PORT")
                code = fr'''
\<script src="https://eros-ai.cloud/aixchat.js"
    data-project-id="{group_id}"
    data-scope-id="{scope_id}"
    data-api-url="https://eros-ai.cloud:{port}"
    data-user-token="35e26211fa1d4746bc814f9cb2a478b8"
    data-div-id="aibot"
\>\<\/script\>
'''.strip()
                logging.info(f'webchat code:\n{code}')
                await self.safe_send_message_to_thread(
                    update,
                   f"To use WebChat add this to your website inside \\<body\\>\n\n```\n{code}\n```",
                   use_markdown=True)

                link = f'https://eros-ai.cloud/chat.html?scopeId={scope_id}&port={port}'
                await self.safe_send_message_to_thread(
                    update,
                    f"WebChat - {link}",
                )
                group_name = await self.get_group_name(group_id)

                await self.safe_send_message_to_thread(update,
                                                       f"🎉 Everything is set up for '{group_name}' #{group_id}!")
            else:
                await self.safe_send_message_to_thread(update, f"Please finish setup first.\n{unfinished_field_name} is missing.")
            return

        elif callback_data.startswith("delete:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            _confirm_key = f'delete-confirm:{group_id}:{user_id}'
            if not self.config.get(_confirm_key):
                self.config.setex(_confirm_key, 'true', 60)
                await self.safe_send_message_to_thread(
                    update,
                    f"Are you sure you want to delete this group? Click again on 'Delete' to confirm!")
            else:
                del self.config[f'setup:{user_id}:group_id']
                del self.config[f'setup:{user_id}:step']
                self.openai.delete_group_settings(group_id)
                await self.safe_send_message_to_thread(update, f"Removed!")

        elif callback_data.startswith("dex_tools_link:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:dex_tools_link']
                await self.safe_send_message_to_thread(update, f'DexTools link:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            await self.safe_send_message_to_thread(update, f"Send full DEXTools link of your token.")
            self.config[f'setup:{user_id}:step'] = 'dex_tools_link'
        elif callback_data.startswith("autoreply:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'enable':
                self.config[f'group_settings:{group_id}:autoreply'] = 'true'
                await self.safe_send_message_to_thread(update, f"Autoreply enabled")
            elif len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'disable':
                self.config[f'group_settings:{group_id}:autoreply'] = 'false'
                await self.safe_send_message_to_thread(update, f"Autoreply disabled")
            else:
                old_value = self.config[f'group_settings:{group_id}:autoreply']
                if not old_value or not is_true(old_value):
                    self.config[f'group_settings:{group_id}:autoreply'] = 'true'
                    await self.safe_send_message_to_thread(update, f"Autoreply enabled")
                else:
                    self.config[f'group_settings:{group_id}:autoreply'] = 'false'
                    await self.safe_send_message_to_thread(update, f"Autoreply disabled")
            await self.send_continue_setup(
                update=update,
                user_id=user_id,
                group_id=group_id,
                prefix=f''
            )
        elif callback_data.startswith("how_to_buy:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:how_to_buy']
                await self.safe_send_message_to_thread(update, f'How to buy:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            await self.safe_send_message_to_thread(update, f"Explain how to buy your token (e.g. send Uniswap link).")
            self.config[f'setup:{user_id}:step'] = 'how_to_buy'

        elif callback_data.startswith("contracts:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:contracts']
                await self.safe_send_message_to_thread(update, f'Contracts:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            await self.safe_send_message_to_thread(update, f"Describe all contracts you use.")
            self.config[f'setup:{user_id}:step'] = 'contracts'

        elif callback_data.startswith("token_info:"):
            group_id = int(callback_data.split(":")[1])
            if group_id != self.config[f'setup:{user_id}:group_id']:
                await self.safe_send_message_to_thread(update,
                                                       f"Wrong group_id! You probably tried to continue old or canceled setup process.")
                return

            if len(callback_data.split(':')) == 3 and callback_data.split(':')[2] == 'get':
                value = self.config[f'group_settings:{group_id}:token_info']
                await self.safe_send_message_to_thread(update, f'Token info:\n{value}')
                await self.send_continue_setup(update, user_id, group_id, prefix=f'')
                return

            await self.safe_send_message_to_thread(update, f"Describe your Token, emission, buy/sell fees etc.")
            self.config[f'setup:{user_id}:step'] = 'token_info'
        else:
            await self.safe_send_message_to_thread(update, f"Unknown button callback")

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                              is_inline=False) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :param is_inline: Boolean flag for inline queries
        :return: Boolean indicating if the user is allowed to use the bot
        """
        if update.effective_chat.type == 'channel':
            name = update.effective_chat.title
            user_id = update.effective_chat.id
        else:
            name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
            user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context, is_inline)
            return False
        if not is_within_budget_v2(self.config, update, is_inline=is_inline, budget_client=self.openai.budget_client):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        return True

    async def send_disallowed_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the disallowed message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.disallowed_message,
                disable_web_page_preview=True
            )
            self.message_queue.enqueue_message(Message(
                chat_id=str(update.effective_chat.id),
                content=self.disallowed_message,
                channel='telegram',
                user_id='ai',
                username='ai',
            ))
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.disallowed_message)

    async def send_budget_reached_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False):
        """
        Sends the budget reached message to the user.
        """
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.budget_limit_message
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(update, result_id, message_content=self.budget_limit_message)

        try:
            _message = f'User {update.message.from_user.username} {update.message.from_user.name} (id: {update.message.from_user.id}) reached their usage limit'
            chat_ids = self.parse_list_from_redis('BUDGET_CHAT_IDS')
            await self.send_message_to_chat_ids(chat_ids, _message)
        except Exception as e:
            pass

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        logging.info(f'post_init')
        me = await application.bot.get_me()
        logging.info(f"Bot started with username: @{me.username}")
        self.bot_id = me.id
        self.config['bot:username'] = me.username

        await application.bot.delete_my_commands(scope=BotCommandScopeAllChatAdministrators())
        await application.bot.set_my_commands(self.group_admin_commands, scope=BotCommandScopeAllChatAdministrators())

        await application.bot.delete_my_commands(scope=BotCommandScopeAllGroupChats())
        await application.bot.set_my_commands(self.group_commands, scope=BotCommandScopeAllGroupChats())

        await application.bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())
        await application.bot.set_my_commands(self.private_commands, scope=BotCommandScopeAllPrivateChats())

    async def process_all_messages_queue(self):
        while True:
            try:
                message = self.message_queue.dequeue_message()
                if not message:
                    await asyncio.sleep(0.05)
                    continue
                if os.environ.get('CLIENT') == 'airdrophunter':
                    if not message.technical:
                        logging.info(f'airdrophunter {message=}')
                        self.airdrophunterdb.enqueue_message(message)
            except Exception as e:
                logging.error(f'Failed to send trigger: {e}')
                logging.exception(e)

    async def _retry_send_message(self, chat_id, text):
        i_try = 0
        while True:
            i_try += 1
            try:
                await self.application.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                if 'Flood control exceeded.' in str(exc):
                    delay = float(str(exc).split('Retry in ')[1].split(' seconds')[0])
                    logging.warning(f'Flood control exceeded, retry in {delay} seconds')
                    await asyncio.sleep(delay + 1)
                    continue
                raise
            break

    async def process_openai_kwargs_queue(self):
        """
        {
            channel: 'telegram',
            username: 'username',
            msg_id: 'msg_id',
            chat_id: 'chat_id',
            openai_kwargs: {...},
            reply: ...
        }

        """
        while True:
            try:
                openai_kwargs_json_raw = self.config.r.rpop('openai_kwargs_queue')
                if not openai_kwargs_json_raw:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logging.exception(f'Failed to get openai_kwargs_queue: {e}')
                await asyncio.sleep(1)
                continue

            try:
                openai_kwargs_json = json.loads(openai_kwargs_json_raw)

                _message = ""
                channel = openai_kwargs_json.get('channel')
                if channel:
                    _message += f"CHANNEL: {channel}\n"
                    _message += f'USERNAME: {openai_kwargs_json.get("username")}\n'
                    if openai_kwargs_json.get('msg_id'):
                        _message += f"MSG_ID: {openai_kwargs_json.get('msg_id')}\n"
                else:
                    if openai_kwargs_json.get('instagram_username'):
                        _message += f"INSTAGRAM_USERNAME: {openai_kwargs_json['instagram_username']}\n"
                        if openai_kwargs_json.get('msg_id'):
                            _message += f"msgId: {openai_kwargs_json.get('msg_id')}\n"
                    elif openai_kwargs_json.get('telegram_username'):
                        _message += f"TELEGRAM_USERNAME: @{openai_kwargs_json['telegram_username'].lstrip('@')}\n"
                        if openai_kwargs_json.get('msg_id'):
                            _message += f"msgId: {openai_kwargs_json.get('msg_id')}\n"
                    elif openai_kwargs_json.get('email_sender'):
                        _message += f"EMAIL_SENDER: {openai_kwargs_json.get('email_sender')}\n"
                    else:
                        logging.error(f'unknown source')
                        continue

                _message += f"CHAT_ID: {openai_kwargs_json['chat_id']}\n"
                _kwargs = {k: v for k, v in openai_kwargs_json['openai_kwargs'].items() if k not in ['messages']}
                _message += f"OPENAI_KWARGS:\n{pprint.pformat(_kwargs)}\n"
                _messages = openai_kwargs_json['openai_kwargs']['messages']
                _conversation = '\n\n'.join([f"{m['role'].upper()}: {m['content']}" for m in _messages])
                _message += f"CONVERSATION:\n{_conversation}\n"
                _message += f"\nREPLY: {openai_kwargs_json['reply']}\n"

                openai_kwargs_chat_ids = self.parse_list_from_redis('OPENAI_KWARGS_CHAT_IDS')
                if not openai_kwargs_chat_ids:
                    logging.info('skip sending openai kwargs')
                await self.send_message_to_chat_ids(openai_kwargs_chat_ids, _message)

                _message = ''
                channel = openai_kwargs_json.get('channel')
                if channel:
                    _message += f"CHANNEL: {channel}\n"
                    _message += f'USERNAME: {openai_kwargs_json.get("username")}\n'
                    if openai_kwargs_json.get('msg_id'):
                        _message += f"MSG_ID: {openai_kwargs_json.get('msg_id')}\n"
                else:
                    if openai_kwargs_json.get('instagram_username'):
                        _message += f"INSTAGRAM_USERNAME: {openai_kwargs_json['instagram_username']}\n"
                        if openai_kwargs_json.get('msg_id'):
                            _message += f"msgId: {openai_kwargs_json.get('msg_id')}\n"
                    elif openai_kwargs_json.get('telegram_username'):
                        _message += f"TELEGRAM_USERNAME: @{openai_kwargs_json['telegram_username'].lstrip('@')}\n"
                    elif openai_kwargs_json.get('email_sender'):
                        _message += f"EMAIL_SENDER: {openai_kwargs_json.get('email_sender')}\n"
                    else:
                        logging.error(f'unknown source')
                        continue

                _message += f"CHAT_ID: {openai_kwargs_json['chat_id']}"
                _messages = openai_kwargs_json['openai_kwargs']['messages']
                _last_user_message = next(_ for _ in reversed(_messages) if _['role'] == 'user') if _messages else {}
                _message += f"\nUSER: {_last_user_message.get('content')}" if _last_user_message else ''
                _message += f"\nREPLY: {openai_kwargs_json['reply']}"
                # _message += f"\nRemaining budget: {openai_kwargs_json['reply']}"

                replies_chat_ids = self.parse_list_from_redis('REPLIES_CHAT_IDS')
                if not replies_chat_ids:
                    logging.info('skip sending reply')
                await self.send_message_to_chat_ids(replies_chat_ids, _message)

            except Exception as e:
                logging.info(f'{openai_kwargs_json_raw=}')
                logging.exception(f'Failed to send trigger: {e}')

            finally:
                await asyncio.sleep(1)

    def parse_list_from_redis(self, key):
        val = self.config[key] or ''
        if val == '-':
            val = []
        elif val:
            val = [int(_) for _ in val.split(',')]
        else:
            val = []
        val = list(set(val))
        return val

    async def send_message_to_chat_ids(self, chat_ids, message):
        if not chat_ids:
            return
        for chat_id in chat_ids:
            try:
                logging.info(f'send_message_to_chat_ids: send to {chat_id}, message: {message[:32]}...')
                await self.send_message_to_chat_id(chat_id, message, technical=True)
            except Exception as e:
                logging.error(f'failed to send to {chat_id}: {e}')

    async def budget_task(self):
        last_at = 0
        while True:
            try:
                # if datetime.datetime.now().minute < 1 and time.time() - last_at > 30 * 60:
                _message = self.get_today_budget_message()
                chat_ids = self.parse_list_from_redis('BUDGET_CHAT_IDS')
                await self.send_message_to_chat_ids(chat_ids, _message)
                last_at = time.time()
            finally:
                await asyncio.sleep(3600)

    async def process_omni_queue(self):
        async def process(data_raw):
            task_kwargs = json.loads(data_raw)
            task = OmniSend(**task_kwargs)
            assert task.channel == 'telegram'
            logging.info(f'OMNI {task.to_dict()}')
            await self.send_message_to_chat_id(chat_id=task.chat_id, text=task.text)

        worker = QueueWorker(
            queue_name=const.OMNI_QUEUE_PREFIX + 'telegram',
            process=process,
            config=self.config,
        )
        await worker.process_forever()

    async def process_triggers_queue(self):
        async def process(data_raw):
            task_kwargs = json.loads(data_raw)
            task = TriggerMessage(**task_kwargs)
            if task.channel == 'instagram':
                task.link = f'https://www.instagram.com/{task.username.lstrip("@")}/'
                task.username = f'@{task.username.lstrip("@")}'
            admin_message_parts = [
                f'Customer is triggered!',
                f'channel: {task.channel}',
                f'username: {task.username}',
                f'chat_id: {task.chat_id}',
            ]
            if task.link:
                admin_message_parts.append(f'link: {task.link}')
            if task.content:
                admin_message_parts.append(task.content)
            trigger_postfix = self.config['TRIGGER_POSTFIX']
            if trigger_postfix:
                admin_message_parts.append(f'{trigger_postfix}')
            admin_message = '\n'.join(admin_message_parts)
            logging.info(f'TRIGGERED {task.channel=} {task.username=} admin_message={admin_message}')
            trigger_chat_ids = self.config.parse_list('TRIGGER_CHAT_IDS')
            await self.send_message_to_chat_ids(trigger_chat_ids, admin_message)

        worker = QueueWorker(
            queue_name='triggers',
            process=process,
            config=self.config,
        )
        await worker.process_forever()

    async def send_message_to_chat_id(
            self,
            chat_id,
            text,
            technical=False,
            reply_markup=None,
            use_markdown=False,
    ):
        chunks = split_into_chunks(text)
        for index, transcript_chunk in enumerate(chunks):
            if reply_markup:
                await self.safe_telegram_request(
                    self.application.bot.send_message,
                    chat_id=chat_id,
                    text=transcript_chunk,
                    reply_markup=reply_markup,
                    parse_mode=constants.ParseMode.MARKDOWN_V2 if use_markdown else None
                )
            else:
                await self.safe_telegram_request(
                    self.application.bot.send_message,
                    chat_id=chat_id,
                    text=transcript_chunk,
                    parse_mode=constants.ParseMode.MARKDOWN_V2 if use_markdown else None
                )
            self.message_queue.enqueue_message(Message(
                chat_id=str(chat_id),
                content=transcript_chunk,
                channel='telegram',
                user_id='ai',
                username='ai',
                technical=technical,
            ))

    async def safe_send_message_to_thread(
            self,
            update,
            text,
            technical=False,
            reply_markup=None,
            use_markdown=False,
    ):
        chat_id = update.effective_chat.id
        chunks = split_into_chunks(text)
        for index, transcript_chunk in enumerate(chunks):
            if reply_markup:
                await self.safe_telegram_request(
                    update.effective_message.reply_text,
                    message_thread_id=get_thread_id(update),
                    text=transcript_chunk,
                    reply_markup=reply_markup,
                    parse_mode=constants.ParseMode.MARKDOWN_V2 if use_markdown else None
                )
            else:
                if use_markdown == 'v1':
                    parse_mode = constants.ParseMode.MARKDOWN
                elif use_markdown:
                    parse_mode = constants.ParseMode.MARKDOWN_V2
                else:
                    parse_mode = None
                await self.safe_telegram_request(
                    update.effective_message.reply_text,
                    message_thread_id=get_thread_id(update),
                    text=transcript_chunk,
                    parse_mode=parse_mode
                )
            self.message_queue.enqueue_message(Message(
                chat_id=str(chat_id),
                content=transcript_chunk,
                channel='telegram',
                user_id='ai',
                username='ai',
                technical=technical,
            ))

    async def safe_reply_to_message(self, update, text, technical=False, reply_markup=None):
        chat_id = update.effective_chat.id
        chunks = split_into_chunks(text)
        _first_sent_msg = None
        for index, transcript_chunk in enumerate(chunks):
            if reply_markup:
                _msg = await self.safe_telegram_request(
                    update.message.reply_text,
                    text=transcript_chunk,
                    reply_markup=reply_markup,
                )
            else:
                _msg = await self.safe_telegram_request(
                    update.message.reply_text,
                    text=transcript_chunk,
                )
            if not _first_sent_msg:
                _first_sent_msg = _msg
            self.message_queue.enqueue_message(Message(
                chat_id=str(chat_id),
                content=transcript_chunk,
                channel='telegram',
                user_id='ai',
                username='ai',
                technical=technical,
            ))
        return _first_sent_msg

    async def receive_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info(f'receive_image, {update=}, {context=}')
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        last_command = self.config.get_last_chat_user_command(chat_id, user_id)
        if last_command != 'ximage':
            logging.info(f'ignore image because {last_command=}')
            return
        await self._ai_image(update, context)

    async def _ai_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.log_command(update.effective_chat.id, update)
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        file_id = update.message.photo[-1].file_id if update.message.photo else update.message.document.file_id
        caption = update.message.caption

        self.config.delete_last_chat_user_command(chat_id, user_id)

        async def _ask():
            try:
                _status, _msg = self.image_ai_limiter.is_allowed_for_user_group(
                    group_id=update.effective_chat.id,
                    user_id=update.message.from_user.id,
                )
                if not _status and not is_admin(self.config, user_id):
                    await self.safe_send_message_to_thread(
                        update=update,
                        text=_msg,
                    )
                    return

                file_path = (await context.bot.getFile(file_id)).file_path
                if not caption:
                    _caption = ''
                elif caption.startswith('/') and caption.count(' ') > 0:
                    _caption = ' '.join(caption.split(' ')[1:])
                else:
                    _caption = caption

                chat_id = update.effective_chat.id
                scope_id = self.get_update_scope_id(update)  # could be None
                if self.config[f'group_settings:{chat_id}:website']:
                    await self.openai.update_assistant_prompt(
                        group_id=chat_id,
                        channel='telegram',
                        _dextools=self.dextools,
                        _erc20prices=self.erc20prices,
                        is_image=True,
                    )

                result, openai_kwargs = await self.openai.reply_on_image(
                    file_path=file_path,
                    caption=_caption,
                    scope_id=scope_id,
                )

                result = await self.post_validations(
                    content=result,
                    context=context,
                    chat_id=chat_id,
                    sent_message=None,
                    openai_kwargs=openai_kwargs,
                    skip_hallucinations=True,
                )

                try:
                    await self.safe_send_message_to_thread(
                        update=update,
                        text=f'{result}',
                        use_markdown='v1',
                    )
                except telegram.error.BadRequest:
                    await self.safe_send_message_to_thread(
                        update=update,
                        text=f'{result}',
                    )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    reply_to_message_id=get_reply_to_message_id(
                        self.config, update),
                    text=const.LIGHT_ERROR_MESSAGE
                )

        await wrap_with_indicator(
            update,
            context,
            _ask(),
            constants.ChatAction.TYPING,
        )

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        logging.info(f"{self.config['PROXY']=}")
        if self.config['PROXY'] and str(self.config['PROXY']).lower() != 'false':
            self.application = ApplicationBuilder() \
                .token(self.config['TELEGRAM_BOT_TOKEN']) \
                .proxy_url(self.config['PROXY'] if self.config['PROXY'] else None) \
                .get_updates_proxy_url(self.config['PROXY'] if self.config['PROXY'] else None) \
                .post_init(self.post_init) \
                .concurrent_updates(True) \
                .build()
        else:
            self.application = ApplicationBuilder() \
                .token(self.config['TELEGRAM_BOT_TOKEN']) \
                .post_init(self.post_init) \
                .concurrent_updates(True) \
                .build()

        self.last_messages_manager = LastMessagesManager(self.config, self.application.bot)

        self.application.add_handler(CommandHandler('reset_budgets', self.reset_budgets))
        self.application.add_handler(CommandHandler('today_budget', self.today_budget))
        self.application.add_handler(CommandHandler('reset', self.reset))
        self.application.add_handler(CommandHandler('reset_scope', self.reset_scope))
        self.application.add_handler(CommandHandler('help', self.help))
        self.application.add_handler(CommandHandler('image', self.image))
        self.application.add_handler(CommandHandler('start', self.start))
        self.application.add_handler(CommandHandler('stats', self.stats))
        self.application.add_handler(CommandHandler('all_stats', self.all_stats))
        self.application.add_handler(CommandHandler('set', self.set))
        self.application.add_handler(CommandHandler('set_wallet', self.set_wallet))
        self.application.add_handler(CommandHandler('noimage', self.noimage))
        self.application.add_handler(CommandHandler('pause', self.pause))
        self.application.add_handler(CommandHandler('unpause', self.unpause))
        self.application.add_handler(CommandHandler('yesimage', self.yesimage))
        self.application.add_handler(CommandHandler('autoreply', self.autoreply))
        self.application.add_handler(CommandHandler('noautoreply', self.noautoreply))
        self.application.add_handler(CommandHandler('whitelist', self.whitelist))
        self.application.add_handler(CommandHandler('mem', self.mem))
        self.application.add_handler(CommandHandler('etherscan', self.etherscan))
        self.application.add_handler(CommandHandler('sql', self.sql))
        self.application.add_handler(CommandHandler('verify', self.verify))
        self.application.add_handler(CommandHandler('signed', self.verify_reply))
        self.application.add_handler(CommandHandler('setup', self.fake_setup))
        self.application.add_handler(CommandHandler('setup_group', self.setup_group))
        self.application.add_handler(CommandHandler('add_file', self.add_file))
        self.application.add_handler(CommandHandler('add_websites', self.add_websites))
        self.application.add_handler(CommandHandler('add_website_to_main_scope', self.add_website_to_main_scope))
        # self.application.add_handler(CommandHandler('add_one_website', partial(self.add_website, only_one=True)))
        self.application.add_handler(CommandHandler('add_website_raw_markdown',
                                                    partial(self.add_website, raw_markdown=True)))
        # self.application.add_handler(CommandHandler('save_website_to_file', partial(self.add_website, to_file=True)))
        # self.application.add_handler(CommandHandler('save_one_website_to_file', partial(self.add_website, to_file=True, only_one=True)))
        self.application.add_handler(CommandHandler('nedviga', self.nedviga))
        self.application.add_handler(CommandHandler('add_website', self.add_website))
        self.application.add_handler(CommandHandler('sff', self.set_from_file))
        self.application.add_handler(CommandHandler('search_vectors', self.search_vectors))
        self.application.add_handler(CommandHandler('search_vectorsf', partial(self.search_vectors, to_file=True)))
        self.application.add_handler(CommandHandler('mems', self.mems))
        self.application.add_handler(CommandHandler('pins', self.pins))
        self.application.add_handler(CommandHandler('pin', self.pin))
        self.application.add_handler(CommandHandler('joke', self.joke))
        self.application.add_handler(CommandHandler('compliment', self.compliment))
        self.application.add_handler(CommandHandler('kb', self.kb))
        self.application.add_handler(CommandHandler('get_scope_id', self.get_scope_id))
        self.application.add_handler(CommandHandler('get', self.get))
        self.application.add_handler(CommandHandler('get_keys_by_pattern', self.get_keys_by_pattern))
        self.application.add_handler(CommandHandler('delete_keys_by_pattern', self.delete_keys_by_pattern))
        self.application.add_handler(CommandHandler('delete', self.delete))
        self.application.add_handler(CommandHandler('exception', self.exception))
        self.application.add_handler(CommandHandler('append', self.append))
        self.application.add_handler(CommandHandler('x', self.x))
        self.application.add_handler(CommandHandler('realtime', self.realtime))
        self.application.add_handler(CommandHandler('ximage', self.ximage))
        self.application.add_handler(CommandHandler('X', self.x))
        self.application.add_handler(CommandHandler('email', self.email))
        # self.application.add_handler(CommandHandler('ai', self.ai))
        # self.application.add_handler(CommandHandler('Ai', self.ai))
        # self.application.add_handler(CommandHandler('AI', self.ai))
        # self.application.add_handler(CommandHandler('aI', self.ai))
        self.application.add_handler(CommandHandler('config', self.get_config))
        self.application.add_handler(CommandHandler('config_to_file', self.get_config_to_file))
        self.application.add_handler(CommandHandler('questions_to_file', self.questions_to_file))
        self.application.add_handler(CommandHandler('c', self.get_config))
        self.application.add_handler(CommandHandler('get_chat_id', self.get_chat_id))
        self.application.add_handler(CommandHandler('add_chat_id_to_REPLIES', self.add_chat_id_to_REPLIES))
        self.application.add_handler(CommandHandler('add_chat_id_to_KWARGS', self.add_chat_id_to_KWARGS))
        self.application.add_handler(CommandHandler('add_chat_id_to_BUDGETS', self.add_chat_id_to_BUDGETS))
        self.application.add_handler(CommandHandler('add_chat_id_to_TRIGGERS', self.add_chat_id_to_TRIGGERS))
        self.application.add_handler(CommandHandler('add_partner', self.add_partner))
        self.application.add_handler(CommandHandler('get_logs', self.get_logs))
        self.application.add_handler(CommandHandler('news_summary', self.news_summary))
        self.application.add_handler(CommandHandler('chat_summary', self.chat_summary))
        self.application.add_handler(MessageHandler(PinnedFilter(), self.handle_pinned_message))
        self.application.add_handler(MessageHandler(PinnedByCommandFilter(self.config),
                                                    self.handle_pinned_message_by_command))
        self.application.add_handler(MessageHandler(AIImageFilter(), self._ai_image))
        self.application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        self.application.add_handler(MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO |
            filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO,
            self.transcribe))
        self.application.add_handler(MessageHandler(
            filters.Document.TEXT, self.receive_text_document))
        self.application.add_handler(MessageHandler(
            filters.Document.IMAGE | filters.PHOTO, self.receive_image))
        self.application.add_handler(MessageHandler(
            filters.Document.ALL, self.receive_text_document))
        self.application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
        ]))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))

        self.application.add_error_handler(error_handler)

        logging.info(f'create task process_triggers_queue')
        asyncio.get_event_loop().create_task(self.budget_task())
        asyncio.get_event_loop().create_task(self.process_triggers_queue())
        asyncio.get_event_loop().create_task(self.process_omni_queue())
        asyncio.get_event_loop().create_task(self.process_openai_kwargs_queue())
        asyncio.get_event_loop().create_task(self.process_all_messages_queue())
        if os.environ.get('CLIENT') == 'airdrophunter':
            logging.info(f'create task airdrophunterdb.airdrophunter_db_writer')
            asyncio.get_event_loop().create_task(self.airdrophunterdb.airdrophunter_db_writer())
        # asyncio.get_event_loop().create_task(self.clients_forever())

        logging.info('run_polling')
        self.application.run_polling(
            poll_interval=0.0,
            timeout=60,
            bootstrap_retries=-1,
            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
            pool_timeout=30,
        )
