import logging
import os

from anthropic import AsyncAnthropic
from telegram import Document
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config import BOT_TOKEN, ANTHROPIC_TOKEN, SUPER_ADMIN_IDS
from database import UserManager

logger = logging.getLogger(__name__)


class Bot:
    def __init__(self, token):
        self.client = AsyncAnthropic(api_key=ANTHROPIC_TOKEN)
        self.token = token
        self.bot = Application.builder().token(token).build()
        self.user_manager = UserManager()
        self.setup_logging()
        self.MAX_TRIES = 3
        self.processing_file = False

    @staticmethod
    def setup_logging():
        logger.setLevel(logging.INFO)

        # Create a FileHandler for writing log file
        fh = logging.FileHandler('bot.log')
        fh.setLevel(logging.INFO)

        # Create a StreamHandler for console output
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Define the handler's output format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)

        _art = r"""
_.~"(_.~"(_.~"(_.~"(_.~"(
_.~"(_.~"(_.~"(_.~"(_.~"(
_.~"(_.~"(_.~"(_.~"(_.~"(
        """

        logger.info(_art)

        logger.addHandler(ch)

    async def claude_response(self, user_prompt, user_id):
        context = self.user_manager.get_context(user_id)
        context += f"{AsyncAnthropic.HUMAN_PROMPT}{user_prompt}{AsyncAnthropic.AI_PROMPT}"
        self.user_manager.update_context(user_id, context)

        try:
            res = await self.client.completions.create(
                prompt=context,
                model="claude-2.1",
                max_tokens_to_sample=200000
            )
            return res
        except Exception as e:
            logger.error(f"Error in claude_response: {e}")
            return None
        finally:
            self.user_manager.update_conversation_count(user_id)

    @staticmethod
    def has_permission(user_id):
        return user_id in SUPER_ADMIN_IDS

    async def handle_message(self, update: Update, _):
        try:
            ob = update.message
            user_id = ob.from_user.id
            chat_type = ob.chat.type
            has_user = self.user_manager.has_user(user_id)

            if ob.text is not None:
                message = ob.text.replace('/c', '')

                if chat_type == 'supergroup' and ob.text.startswith('/c'):
                    if has_user and message.strip() != '':
                        await self.process_conversation(message, user_id, update)
                    elif not has_user:
                        await update.message.reply_text('Provide your user ID by using -> @myidbot.')
                    elif message.strip() == '':
                        await update.message.reply_text("Message is Empty")
                elif chat_type == 'private' and self.user_manager.has_user(user_id):
                    if message.strip() == '':
                        await update.message.reply_text("Message is Empty")
                    elif message.strip() != '':
                        await self.process_conversation(message, user_id, update)
                    elif not has_user:
                        await update.message.reply_text('Provide your user ID by using -> @myidbot.')
            elif ob.document and chat_type == 'private' and self.user_manager.has_user(user_id):
                await self.process_file_conversation(ob.document, user_id, update)

        except Exception as e:
            logger.error(f"Error in handle_message: {e}")

    async def process_file_conversation(self, document: Document, user_id, update):
        if self.processing_file:
            await update.message.reply_text("Another file is being processed. Please wait.")
            return

        file_name = document.file_name
        file_path = os.path.join(r'/root/claude/', file_name)

        try:
            if 30000000 > update.message.document.file_size > 5 and (
                    file_name.endswith('.txt') or file_name.endswith('.md')):
                self.processing_file = True  # 设置处理文件标志为 True

                doc = await update.message.document.get_file()
                await doc.download_to_drive(file_path)

                # 处理文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                if not file_content:
                    await update.message.reply_text("Empty file. Please upload a non-empty text file.")
                    return

                await self.process_conversation(file_content, user_id, update)

            else:
                await update.message.reply_text("Not a TXT file or file size less than 5 bytes.")

        except Exception as e:
            logger.error(f"Error in process_file_conversation: {e}")
        finally:
            os.remove(file_path)
            self.processing_file = False  # 重置处理文件标志为 False

    async def process_conversation(self, message, user_id, update):
        try:
            if self.user_manager.get_conversation_count(user_id) >= 50:
                self.user_manager.clear_context(user_id)
                self.user_manager.clear_conversation_count(user_id)

            tries = 0
            message_text = None

            while tries < self.MAX_TRIES and not message_text:
                response = await self.claude_response(message, user_id)
                if response:
                    message_text = response.completion
                else:
                    logger.error("Failed to get response from Claude")
                tries += 1

            if message_text:
                if len(message_text) < 2000:
                    await update.message.reply_text(message_text)
                else:
                    with open('/root/claude/result.txt', 'w', encoding='utf-8') as f:
                        f.write(message_text)
                    await update.message.reply_document('/root/claude/result.txt')
                    os.remove('/root/claude/result.txt')
            else:
                await update.message.reply_text("Empty response from Claude")
        except Exception as e:
            logger.error(f"Error in process_conversation: {e}")

    async def handle_clear(self, update: Update, _):
        try:
            user_id = update.message.from_user.id
            has_user = self.user_manager.has_user(user_id)
            if has_user:
                self.user_manager.clear_context(user_id)  # Clear the context in the database
                await update.message.reply_text("Context cleared successfully.")
        except Exception as e:
            logger.error(f"Error in handle_clear: {e}")

    async def handle_list(self, update: Update, _):
        try:
            if update.message.from_user.id in SUPER_ADMIN_IDS:
                users = self.user_manager.get_users()
                user_list = '\n'.join(str(user) for user in users)
                await update.message.reply_text(f"`{user_list}`", parse_mode="MarkdownV2")
        except Exception as e:
            logger.error(f"Error in handle_list: {e}")

    async def handle_search(self, update: Update, context):
        try:
            if update.message.from_user.id in SUPER_ADMIN_IDS:
                user_id = int(context.args[0])
                user = self.user_manager.get_user(user_id)

            if 2000 > len(str(user)) > 0:
                await update.message.reply_text(str(user))
            elif len(str(user)) > 2000:
                with open('/root/claude/query.log', 'w', encoding='utf-8') as f:
                    f.write(str(user))
                await update.message.reply_document('/root/claude/query.log')
                os.remove('/root/claude/query.log')
            else:
                await update.message.reply_text(f"User not found with ID: {user_id}")

        except Exception as e:
            logger.error(f"Error in handle_search: {e}")

    async def handle_delete(self, update: Update, context):
        try:
            if update.message.from_user.id in SUPER_ADMIN_IDS:
                user_id = int(context.args[0])
                deleted = self.user_manager.delete_user(user_id)
                if deleted:
                    await update.message.reply_text(f"User deleted with ID: {user_id}")
                else:
                    await update.message.reply_text(f"User not found with ID: {user_id}")
        except Exception as e:
            logger.error(f"Error in handle_delete: {e}")

    async def handle_add(self, update: Update, context):
        try:
            if update.message.from_user.id in SUPER_ADMIN_IDS:
                user_id = int(context.args[0])
                self.user_manager.add_user(user_id)
                await update.message.reply_text(f"User added with ID: {user_id}")
        except Exception as e:
            logger.error(f"Error in handle_add: {e}")

    @staticmethod
    async def handle_log(update: Update, _):
        try:
            if update.message.from_user.id in SUPER_ADMIN_IDS:
                try:
                    with open('bot.log', 'r') as file:
                        lines = file.readlines()
                        last_lines = ''.join(lines[-20:])
                        await update.message.reply_text(rf'`{last_lines}`', parse_mode="MarkdownV2")
                except Exception as e:
                    logger.error(f"Error while reading log file: {e}")
                    await update.message.reply_text("An error occurred while reading the log file.")
        except Exception as e:
            logger.error(f"Error in handle_log: {e}")

    async def handle_start(self, update: Update, _):
        _start = r"""
︻╦╤─  

︻デ═一

╦̵̵̿╤─ ҉ ~ •
        """
        try:
            user_id = update.message.from_user.id
            has_user = self.user_manager.has_user(user_id)
            if has_user:
                await update.message.reply_text(_start)
        except Exception as e:
            logger.error(f"Error in handle_start: {e}")

    def run(self):
        try:
            self.bot.add_handler(CommandHandler("c", self.handle_message))
            self.bot.add_handler(CommandHandler("clear", self.handle_clear))
            self.bot.add_handler(CommandHandler("list", self.handle_list))
            self.bot.add_handler(CommandHandler("search", self.handle_search))
            self.bot.add_handler(CommandHandler("delete", self.handle_delete))
            self.bot.add_handler(CommandHandler("add", self.handle_add))
            self.bot.add_handler(CommandHandler("log", self.handle_log))
            self.bot.add_handler(CommandHandler("start", self.handle_start))
            self.bot.add_handler(MessageHandler(filters.TEXT | filters.ATTACHMENT, self.handle_message))
            self.bot.run_polling()
        except Exception as e:
            logger.error(f"Error in run: {e}")


if __name__ == '__main__':
    bot = Bot(BOT_TOKEN)
    bot.run()
