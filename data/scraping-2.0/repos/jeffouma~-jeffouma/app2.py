import telebot
import openai

# Set up your OpenAI API credentials
openai.api_key = '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'

# Initialize the Telegram bot
BOT_TOKEN = "6370563437:AAHrFiuPUeVhnKNVMtNfHmX9UEWROvZIPs0"
bot = telebot.TeleBot(BOT_TOKEN)

# Initialize conversation history as an empty list
conversation_history = []

# Function to send a message to ChatGPT and get a response
def chat_with_gpt(prompt):
    # Include the entire conversation history as the prompt to maintain context
    conversation_prompt = "\n".join(conversation_history)
    prompt = f"{conversation_prompt}\nUser: {prompt}\n"

    # Send user prompt to ChatGPT
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None
    )

    # Extract the generated reply from the API response
    reply = response.choices[0].text.strip()
    return reply

# Handler for all incoming messages
@bot.message_handler(func=lambda msg: True)
def handle_message(message):
    global conversation_history

    # Get the user's message
    user_input = message.text

    # Add the user's message to the conversation history
    conversation_history.append(f"User: {user_input}")

    # Check if the user's message is a reply to the bot's message
    if message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id:
        # Extract the bot's previous response and add it to the conversation history
        previous_bot_response = message.reply_to_message.text
        conversation_history.append(f"Bot: {previous_bot_response}")

    # Generate a response from ChatGPT based on the user's input and context
    response = chat_with_gpt(user_input)

    # Add the bot's response to the conversation history
    conversation_history.append(f"Bot: {response}")

    # Send the response to the user
    bot.reply_to(message, response)

# Start the bot
bot.polling()

