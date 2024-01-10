
import os
from dotenv import load_dotenv

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram import Update
from telegram.utils.helpers import escape_markdown

import openai
# import config
import logging
from logging.handlers import TimedRotatingFileHandler
import re
import sys



load_dotenv()
openai.api_key =  os.getenv("OPENAI_KEY")


# set environment
try :
    ## TODO: TEST database

    environment = sys.argv[1]
    print("Environment: ", environment)
    
    if environment == "test":
        telegram_token=os.getenv("TELEGRAM_TEST_TOKEN")
    elif environment == "prod":
        telegram_token=os.getenv("TELEGRAM_TOKEN")
    else:
        print("provide argument: test or prod")
        sys.exit(1)
except Exception as error:
    print("Error: ", error)
    print("\n\nProvide argument: test or prod")
    sys.exit(1)


## Settings for logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('my_logger')
handler = TimedRotatingFileHandler(filename=f'logs/{environment}/t.log', 
                                   when='midnight', interval=1, backupCount=0)
handler.suffix = '_%Y-%m-%d.log'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



users={}
max_tokens=1800

def start(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="Hi! I'm your AI assistant. Ask me anything and I'll respond within a few seconds!")
    logger.info(f'User {update.message.from_user.username} started the bot')

def respond(update, context):
    message = cleanup_text(update.message.text)
    name = update.message.from_user    
    
    last_name = name.last_name if name.last_name else ""
    logger.info(f"User {name.username} ({name.first_name} {last_name}, {name.id}), Message: \"{message}\" ")
    
    if get_word_count(message) < max_tokens/2:
        ## if message is short enough, send to openAI
    
        print("\n\nPrompt-----------------------")
        print(name)
        print()
        print(message)
        
        ## get messages log for user. Initialize as blank if needed.
        username = name['username']
        messages = users.get(username, []) 
        if(len(messages) == 0):
            users[username]=messages

        new_message = {"role": "user", "content": message}
        # print("\nnew_message: ", new_message)
        
        ## add newest message to messages log. length capped at 10
        add_to_chatlog(username, new_message)

        try:
            ## The API Call: get reseponse from openAI.
            completions = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=0.7,
            )
        except Exception as e:
            print("Exception: ", e)
            context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I'm having trouble responding. Can you try again?")
            try:
                print("Log warning")
                logger.warning("Error generating response from OpenAI: ", e)
            except Exception as err:
                print("Error logging warning")
                print(err)

            return
        
        total_tokens_used = completions['usage']['total_tokens']
        print("\ntotal_tokens_used: ", total_tokens_used)
        try:
            logger.info(f"tokens_used by {username}: {total_tokens_used}")
        except Exception as e:
            print("error logging tokens_used")
            print(e)


        # if total_tokens_used <= max_tokens:
        try:
            generated_text = completions.choices[0]["message"]["content"]
            generated_text_oneline = generated_text.replace("\n", " ")
            logger.info(f"Generated Response: \"{generated_text_oneline[:100]}\" ")
        except Exception as e:
            print("Error logging response")
            print(e)
        
        print("\nResponse---------------------")
        print(generated_text)
        
        generated_text_clean = cleanup_text(generated_text)
        
        try:
            context.bot.send_message(chat_id=update.effective_chat.id, text=generated_text_clean, parse_mode="Markdown")
            logger.info(f"Sent response to {username}")
        except Exception as e:
            print("Exception: ", e)
            try:
                logger.warning("Error sending response to Telegram: ", e)
                context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, something weird happened. Can you try again?", parse_mode="Markdown")
            except Exception as e:
                print("Error logging warning")
                print(e)
            return
        
        
        add_to_chatlog(username, {"role": "assistant", "content": generated_text_clean})
       
        print("\n\nActive users: ", users.keys())
        print()
    else:
        alert_text="That message was too long for me to process. Can you try to shorten it to about 1000 words?"
        print("\n", alert_text, "\n")
        context.bot.send_message(chat_id=update.effective_chat.id, 
                                 text=alert_text, 
                                 parse_mode="Markdown")



def cleanup_text(text):
    paragraphs = text.split("\n\n") 


    for p in paragraphs:
        if re.match(r"^ {4}", p):
            p = "```\n" + p.lstrip() + "\n```" 

    updated_text = "\n\n".join(paragraphs)
    
    return updated_text

def add_to_chatlog(username, message):
    user_messages = users[username]
    user_messages.append(message)
    
    ## delete the first 2 messagelogger.infos if there are more than 10
    if len(user_messages) >10:
        logger.info(f"Removing older message from chat log for {username}")
        print("\nRemoving older message from chat log...")
        user_messages = user_messages[1:]

    ## setting max size to half of max_tokens. Keep removing first element whie size is greater than max_tokens/2
    word_count = count_words_in_array(user_messages)
    print("\nword_count: ", word_count)
    
    while word_count > max_tokens/2:
        print("\nRemoving older message from chat log...")
        user_messages = user_messages[1:]
        word_count = count_words_in_array(user_messages)
        print("\nword_count: ", word_count)
    
    users[username] = user_messages
    logger.info(f"Chat log for {username} now has {len(user_messages)} messages")

def count_words_in_array(messages):
    word_count = 0
    for message in messages:
        text = message.get('content', '')
        words = text.split()
        word_count += len(words)
    return word_count

def get_word_count(string):
    words = string.split()
    return len(words)

def main():
    updater = Updater(token=telegram_token, use_context=True)
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)
    respond_handler = MessageHandler(Filters.text, respond)
    dispatcher.add_handler(respond_handler)
    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':

    
    logger.info(f'Started app')

    main()