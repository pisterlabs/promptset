# Import the necessary modules
import openai
import telegram
from telegram.ext import Updater, CommandHandler

# Set the API tokens and initialize the bot
tg_Token = "<set token>"
openai.api_key = "<set token>"
bot = telegram.Bot(token=tg_Token)


# Define a function to send a welcome message when the /status command is used
def status(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Hello HKers! I'm a ChatGPT bot!")
    context.bot.send_message(chat_id=update.message.chat_id, text="To ask a question to ChatGPT, please use /askcgpt <Msg>")


# Define a function to handle the /askcgpt command and send a message to ChatGPT
def askChatGPT(update, context):

    # Send a message indicating the question has been received and is being processed
    update.message.reply_text("The question has been sent to ChatGPT...")

    # Get the message text and define the model to use
    question = update.message.text
    model_name="gpt-3.5-turbo"

    # Create a conversation object and use the OpenAI API to get a response
    conversation = [{'role':'user','content':question}]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=conversation
    )

    # Send the response back to the user
    update.message.reply_text(response['choices'][0]['message']['content'])


# Create an updater object and add the command handlers
updater = telegram.ext.Updater(token=tg_Token, use_context=True)
updater.dispatcher.add_handler(CommandHandler('askcgpt', askChatGPT))
updater.dispatcher.add_handler(CommandHandler('status', status))

# Start the bot and listen for messages
updater.start_polling()
updater.idle()
