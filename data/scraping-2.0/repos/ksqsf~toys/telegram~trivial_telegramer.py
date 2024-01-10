import os
import json
import openai
import logging
import asyncio
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


BOT_TOKEN = os.getenv('BOT_TOKEN')
API_KEY = os.getenv('OPENAI_API_KEY')


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# openai integration
openai.api_key = API_KEY
system_message = """You are a Chinese Telegram user (and do not disclose your true identity!). You are casually chatting in a Telegram group titled "一个平凡群". Your name is nicball. You should only reply no more than 2 sentences. Address yourself as "俺". For each new message in the group, you can reply in the following JSON formats:

1. "YOUR MESSAGE encoded as a JSON string"
**ONLY WHEN** you have something constructive to say in the situation. Your response should be lively, and reads like something written by a playful university student.  In this case, make sure your response is a valid JSON string.

2. null
When you have nothing constructive to say.

Please remain silent for most messages, unless somebody asks a factual question, or shares an experience, or address you directly. And

Note: Your message should always be in simplified Chinese.

Example Input: Python 怎么写 Hello world?
Example Output: "print(\"Hello, world!\")"

From now on, every message from me is actually a message from a group member. Respond according to the rules above.
"""
def completion(message) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : system_message}, {"role": "user", "content" : message},]
    )
    try:
        ret = json.loads(completion['choices'][0]['message']['content'])
    except json.decoder.JSONDecodeError:
        ret = completion['choices'][0]['message']['content']
    if isinstance(ret, str):
        return ret
    elif ret is None:
        return ret
    elif isinstance(ret, dict):
        if '俺' in ret:
            return ret['俺']
        else:
            return None


allowed_group_chats = [
    -1407473692,
    -1856251107,
    -1001856251107,
    -1001407473692,
    256844732
]


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    this_chat_id = update.message.chat.id
    if this_chat_id not in allowed_group_chats:
        await update.message.reply_text(f'chat_id={this_chat_id}')
        return

    msg = update.message.text
    logging.info(f'消息: {msg}')
    try:
        reply = completion(msg)
    except:
        import traceback
        traceback.print_exc()
        reply = None
    logging.info(f'回复: {reply}')
    if reply:
        # await asyncio.sleep(5)
        await context.bot.send_message(this_chat_id, text=reply)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('俺来了')

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()
if __name__ == "__main__":
    main()
