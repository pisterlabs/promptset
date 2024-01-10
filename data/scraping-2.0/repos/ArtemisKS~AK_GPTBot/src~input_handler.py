import openai
import logging
import threading
from typing import List
from telegram.ext import CallbackContext
from telegram import Update, Message
import asyncio
from telegram.error import RetryAfter

from admin_menu_manager import AdminMenuManager
from financial_validator import FinancialValidator
from message_limit_handler import MessageLimitHandler
from translator import Translator
from localization import loc

class InputHandler:
    """
    A class for handling input for GPTBot other than commands.
    """
    def __init__(self, message_limit_handler, financial_validator, updater):
        self.chat_states = {}  # Add this line
        self.message_limit_handler: MessageLimitHandler = message_limit_handler
        self.financial_validator: FinancialValidator = financial_validator
        self.updater = updater
        self.translator = Translator()
        self.admin_menu_manager = AdminMenuManager(message_limit_handler, financial_validator)
        
        # Create an event to stop the typing action when the response is received
        self.stop_typing_event: threading.Event = None
        
        self.total_tokens_used = 0
        
    def admin_notifications_enabled(self, chat_id: int) -> bool:
        self.admin_menu_manager.admin_notifications_enabled(chat_id)
        
    def get_admin_notification_chat_id(self, chat_id: int):
        self.admin_menu_manager.get_admin_notification_chat_id(chat_id)
        
    def show_admin_menu(self, update: Update, context: CallbackContext):
        self.admin_menu_manager.show_admin_menu(update, context)
            
    def handle_admin_callback(self, update: Update, context: CallbackContext):
        self.admin_menu_manager.handle_admin_callback(update, context)
        
    def handle_text(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        
        if user_id in context.chat_data and context.chat_data[user_id]:
            del context.chat_data[user_id]
            self.process_gpt_request(context, update.message, update.effective_chat.id)
        else:
            self.admin_menu_manager.handle_text(update, context)
        
    def start_gpt_question(self, update: Update, context: CallbackContext):
        user_id = update.effective_user.id
        context.chat_data[user_id] = True
        update.message.reply_text(f'{loc("enter_question")}:')
            
    def send_typing_action(self, chat_id, stop_typing_event, context):
        if self.stop_typing_event is None:
            return
        while not stop_typing_event.is_set():
            context.bot.send_chat_action(chat_id=chat_id, action='typing')
            stop_typing_event.wait(5)

    def notify_admins_limit_reached(self, chat_id: int, limit_type: str, context: CallbackContext):
        if not self.admin_menu_manager.admin_notifications_enabled(chat_id):
            return

        dest_chat_id = self.admin_menu_manager.get_admin_notification_chat_id(chat_id)
        if dest_chat_id:
            chat_name = self.get_chat_name(context, chat_id)
            message = loc("limit_reached", limit_type=limit_type, chat_name=chat_name)
            self.updater.bot.send_message(chat_id=dest_chat_id, text=message)
        else:
            message = loc("no_destination_chat_id", chat_id=chat_id)
            logging.info(message)
            
    def get_chat_name(self, context: CallbackContext, chat_id: int):
        bot = context.bot
        chat = bot.get_chat(chat_id)
        return chat.title or chat.username
    
    def start_typing(self, context: CallbackContext, chat_id: int):
        self.stop_typing_event = threading.Event()
        # Start a separate thread to send the typing action
        typing_thread = threading.Thread(target=self.send_typing_action, args=(chat_id, self.stop_typing_event, context))
        typing_thread.start()
        
    def stop_typing(self):
        if self.stop_typing_event is None:
            return
        self.stop_typing_event.set()
        
    async def send_message_with_delay(self, context, chat_id, text, delay):
        await asyncio.sleep(delay)
        context.bot.send_message(chat_id=chat_id, text=text)
    
    def process_gpt_request(self, context: CallbackContext, message: Message, chat_id: int):
        self.chat_states[chat_id] = None  # Reset the chat state

        if not self.is_request_allowed(message, chat_id):
            return

        question = message.text
        
        self.start_typing(context, chat_id)

        bot_system_desc = self.admin_menu_manager.get_bot_description(chat_id)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": bot_system_desc},
                    {"role": "user", "content": f"{question}\n\nAnswer:"}
                ])
        except openai.error.RateLimitError:
            message.reply_text(loc('model_overloaded'))
            self.stop_typing()
            return
        except Exception as e:
            logging.error(f"An error occurred while processing the GPT request: {e}")
            message.reply_text(loc('gpt_error_message'))
            self.stop_typing()
            return

        self.handle_gpt_response(chat_id, context, response)

        # Set the event to stop the typing action
        self.stop_typing()

        answer_text = response.choices[0].message.content.strip()
        try:
            message.reply_text(answer_text)
        except RetryAfter as e:
            logging.warning(f"RetryAfter error, waiting {e.retry_after} seconds before sending message")
            asyncio.ensure_future(self.send_message_with_delay(context, chat_id, answer_text, e.retry_after))
        except Exception as e:
            logging.error(f"An error occurred while sending the GPT response: {e}")
            message.reply_text(loc('gpt_error_message'))

    def is_request_allowed(self, message: Message, chat_id: int) -> bool:
        if not self.financial_validator.can_send_message(chat_id):
            message.reply_text(loc('daily_usd_limit_reached'))
            return False
        elif not self.message_limit_handler.can_send_message(chat_id):
            message.reply_text(loc('daily_limit_reached'))
            return False
        return True

    def handle_gpt_response(self, chat_id: int, context: CallbackContext, response):
        tokens_used = response["usage"]["total_tokens"]
        self.total_tokens_used += int(tokens_used)
        logging.info(f'{tokens_used} tokens used; {self.total_tokens_used} total tokens used (since bot launch) == {self.financial_validator.calculate_usd(self.total_tokens_used)}$')

        # Register the message in the MessageLimitHandler
        self.message_limit_handler.register_message(chat_id)
        # Register the message in the FinancialValidator
        self.financial_validator.register_tokens(chat_id, tokens_used)

        if not self.financial_validator.is_spending_within_limit(chat_id):
            self.notify_admins_limit_reached(chat_id, "USD", context)
        elif not self.message_limit_handler.is_within_message_limit(chat_id):
            self.notify_admins_limit_reached(chat_id, "messages", context)
    
    def gpt(self, update: Update, context: CallbackContext, args: List[str]):
        if args and len(args) > 0:
            # Get the appropriate message object
            message = update.edited_message if update.edited_message != None else update.message

            self.process_gpt_request(context, message, update.effective_chat.id)
        else:
            self.start_gpt_question(update, context)
            
    def is_user_admin(self, update: Update, context: CallbackContext) -> bool:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        return self.admin_menu_manager.is_user_admin(user_id, chat_id, context)