from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import datetime
import json
import os
import openai

updater = Updater("MY_TELE_BOT_ID",
                  use_context=True)
  
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
       "Hello, welcome to the COVID-19 response bot. Please enter your questions / concerns to allow us to direct you to the right channels.")
    print('user started the bot!')
    

def help(update: Update, context: CallbackContext):
    #tells the user what the bot is about and what they can do with it
    update.message.reply_text("This bot can be used for questions / queries regarding COVID-19.")

  
def unknown(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Sorry '%s' is not a valid command" % update.message.text)

def texts(update: Update, context: CallbackContext):
    #receive text input from user and save it as text
    text = update.message.text.lower()

    openai.api_key = 'MY_OPENAI_KEY' # my openai key

    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Classify the urgency of a message as 0,1,2,3\n\nDefinition: Spam, nonsense messages. Unrelated to health issues.\nMessages: What is your favorite color? / What is your favorite hobby? / What is your favorite thing to do?\nUrgency: 0 \n\n##\n\nDefinition: User needs help but it is not time-critical at all, users can read the answers themselves on the Ministry of Health website\nMessages:  What is covid / What is the number of cases today / What is the covid situation in Singapore now / When is the covid situation stabilising in Singapore? / Do we still have to wear a mask? / Is covid still a problem in Singapore? /What are symptoms of Covid? /What is the death rate from covid19? /How can I find out if I have Covid19/ How can I protect myself from Covid 19? / I have no ART kit, where can i get one? / When will covid be endemic? / What is the incubation period?  / Are booster jabs still needed? \nUrgency: 1\n\n##\n\nDefinition: User needs help but help can be provided through an automated chatbot / text\nMessages: I have covid, can I still go out? / I have covid but I share room with my brother where do I go / I am coughing, what should I do? / I am ART positive, what do I do right now? / My nearest clinic is not open, how now? / My legs ache  / How long will I have to quarantine? / What are the long-term effects of covid? / What are the chances of me dying from covid? / I don't have enough medicine / I am coughing and have no medication / I need anti-viral medication / A family member is sick with Covid at home, can I still go to work? \nUrgency: 2\n\n##\nDefinition: User has serious issues that require the immediate attention of a call centre\nMessages: I am sick and I am think I am dying.. / I am feeling very weak.. where can i go for help for covid /My throat hurts really badly.. Where do I go? /I have covid but my boss asked me to go to work.. please help../ My father has covid but he’s still at home.. When is MOH coming to get him to the isolation facility? \nUrgency: 3\n \nMessage: " + text + " \nUrgency: ",
      temperature=0.7,
      max_tokens=2000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    #to interpret the data and determine the urgency based on the input text
    response_dict = json.loads(str(response))
    urgency = int(response_dict['choices'][0]['text'])

    if urgency == 0:
        update.message.reply_text("Please enter a valid message\neg. What are the symptoms of Covid?")

    elif urgency == 1:
        update.message.reply_text("You will be able to seek the answer to this query on the MOH website at \nmoh.gov.sg/faqs")

    elif urgency == 2:
        update.message.reply_text("Please seek further help at the chatbot below \nt.me/covidhelp")  #dummy bot work in progress

    elif urgency == 3:
        update.message.reply_text("Please hold on and do not leave your house. An assistant would be calling you shortly.")
        
        
def unknown_text(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Sorry I can't recognize you , you said '%s'" % update.message.text)
  

#to handle incoming commands
updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(MessageHandler(Filters.command, unknown))  # Filters out unknown commands

#line to deal with texts i
updater.dispatcher.add_handler(MessageHandler(Filters.text, texts))
  
# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

#start running the bot proper
updater.start_polling()


