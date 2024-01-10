import telegram
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler

# Import OpenAI library for ChatGPT
import openai

# Set up your Telegram Bot API token
TELEGRAM_TOKEN = '6967743469:AAGUviYWycx9JWS7inqzVYJx0g61AiQZ8R0'

# Set up your OpenAI API key
OPENAI_API_KEY = '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'

# Initialize your ChatGPT OpenAI API client
openai.api_key = OPENAI_API_KEY

# Dictionary to store conversation history
conversation_history = {}

# Function to handle incoming messages
def handle_message(update, context):
    user_message = update.message.text
    chat_id = update.message.chat_id

    # Get or initialize conversation history for this chat_id
    history = conversation_history.get(chat_id, [])

    # Add the user's message to the conversation history
    history.append({"role": "user", "content": user_message})
    conversation_history[chat_id] = history

    # Use the entire conversation history as context for ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}] + history
    )

    # Get the response from ChatGPT
    bot_response = response['choices'][0]['message']['content']

    # Add ChatGPT's response to the conversation history
    history.append({"role": "system", "content": bot_response})
    conversation_history[chat_id] = history

    # Send ChatGPT's response back to the user
    context.bot.send_message(chat_id=chat_id, text=bot_response)

# Function to handle /start command
def start(update, context):
    chat_id = update.message.chat_id
    context.bot.send_message(chat_id=chat_id, text="Welcome! I'm your ChatGPT bot. Feel free to start a conversation.")

# Function to handle /help command
def help(update, context):
    chat_id = update.message.chat_id
    help_text = "I'm a ChatGPT bot. Just start chatting with me and I'll respond!"
    context.bot.send_message(chat_id=chat_id, text=help_text)

def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Handle incoming messages with the handle_message function
    message_handler = MessageHandler(Filters.text & ~Filters.command, handle_message)
    dispatcher.add_handler(message_handler)

    # Handle /start command
    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    # Handle /help command
    help_handler = CommandHandler('help', help)
    dispatcher.add_handler(help_handler)

    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
