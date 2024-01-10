import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sqlite3_db.utils import init_db, save_to_db, query_todays_chat_history
from langchain_agent.chat import ChatBot

TOKEN = os.environ.get('telegram_refugees_ukr_ch_bot')
BOT_NAME = os.environ.get('telegram_bot_name')

# Create instance of ChatBot class
chatbot = ChatBot()

async def start_command(update: Update, context: ContextTypes):
    await update.message.reply_text('Hi! I am a chatbot that can answer your questions about refugees in Switzerland. Ask me anything!')

async def help_command(update: Update, context: ContextTypes):
    await update.message.reply_text('Please question about aslum or migration in Switzerland. I will try my best answer it.')

async def custom_command(update: Update, context: ContextTypes):
    await update.message.reply_text('This is a custom command, you can add whatever text you want here.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type = update.message.chat.type
    user_id = update.message.from_user.id

    text = update.message.text
    
    chat_history = query_todays_chat_history(user_id=user_id)

    if message_type == 'group':
        if BOT_NAME in text:
            new_text = text.replace(BOT_NAME, "").strip()
            result= chatbot.chat(new_text, chat_history)
        else:
            return
    else:
        result = chatbot.chat(text, chat_history)

    await update.message.reply_text(result["answer"])

    # Save interaction to the database
    save_to_db(user_id=user_id, 
                     user_message=chatbot.translate_to_english_deepL(text),
                     user_message_rephrased=chatbot.translate_to_english_deepL(result["generated_question"]), 
                     bot_response=chatbot.translate_to_english_deepL(result["answer"]), 
                     top_k_docs=result['source_documents'])

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print("Initializing DB...")
    init_db()
    print("Starting bot...")

    app = Application.builder().token(TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    print("polling...")
    # Polls the telegram server for updates
    app.run_polling(poll_interval=0.5)

