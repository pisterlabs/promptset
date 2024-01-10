import telebot
import openai
import yaml

openai.api_key = "Your openai api key"
bot = telebot.TeleBot("Your telegram bot token")
# The telegram user id who authorized to use this bot, leave empty if everyone is allowed.
AUTHORIZED_USER_IDS = []

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        temperature=0.9,)
    return response["choices"][0]["text"]

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "This is a chatgpt3 language model, let's talk.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    user_id = message.from_user.id
    print("User: ", message.from_user.username, "\nText:", text)
    if len(AUTHORIZED_USER_IDS) > 0 and user_id not in AUTHORIZED_USER_IDS:
        bot.send_message(chat_id=message.chat.id, text="User not authorized userid:" + str(user_id))
    else:
        try:
            response = generate_response(text)
            print("\nResponse:", response, "\n")
            bot.send_message(chat_id=message.chat.id, text=response)
        except Exception as e:
            print(traceback.format_exc())
            bot.send_message(chat_id=message.chat.id, text=traceback.format_exc())

if __name__=="__main__":
    bot.polling()
