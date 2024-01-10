from telegram.ext import *
import openai

token= 'TEL-API'
openai.api_key = "GPT-API"
chatlist=[]
username=""

prefix = input("Enter Your Prefix to the AI: ")
chatlist.append(prefix)

def sample_responses(input_text,username):
    username=username
    print(username)
    user_message = str(input_text).lower()
    joinedlist=",".join(chatlist)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{joinedlist} {username}:{user_message}AI:", #plugged AI at the end to make it stop typing
        temperature=0.9,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[f"{username}:","AI:"]) # remove the AI: to if there is a prefix
    chatlisnl = response.choices[0].text
    chatlist.append(f"{username}:{user_message}")
    chatlist.append(f"AI:{chatlisnl}")
    print (joinedlist)
    print (response.choices[0])
    return response.choices[0].text.replace(',', '').replace('AI:', '')
    
def start(update, context):
    update.message.reply_text('Hi!')
    
def handle_message(update, context):
    username = update.message.from_user.first_name
    text = str(update.message.text).lower()
    response = sample_responses(text, username)
    update.message.reply_text(response)

def error(update, context):
    print(f"Update {update} caused error {context.error}")

def main():
    print("Bot started...")
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_error_handler(error)

    updater.start_polling()
    updater.idle()

main()
