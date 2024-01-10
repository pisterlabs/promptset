#------------------------------------------------------------------libs-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
from keras.models import load_model 
from PIL import Image, ImageOps  
import numpy as np
from typing import Final
import telebot
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import time
from caloriefinder import get_nutrient_info 
import random
from telebot import types
import pyodbc
import mysql.connector
from telebot.types import Message, Location,InlineKeyboardMarkup, InlineKeyboardButton
from conditions import conditions
import wikipedia
import requests
from bs4 import BeautifulSoup
import openpyxl
import schedule
import time
from sklearn.ensemble import RandomForestClassifier
from sympt import symptoms_to_diseases
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import threading
import datetime
import subprocess
import bot
print("main file started...")
from bot import bot 
# Use the bot instance in your code specific to reminder.py
# Rest of the code in reminder.py
import openai
from gpt import ask_gpt
import speech_recognition as sr
import tempfile
import os
import json
import base64
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from recomend import get_health_articles

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
print("main file starting...")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
TOKEN = '6072587014:AAF9h6Ji1dRyC9yEL1u7UhAKOevbaKVpyPk'

bot2 = telebot.TeleBot('TOKEN')

BOT_USERNAME: Final = '@werus_co_bot'
print('Starting up bot...')
print('bot is online ')
# API endpoint for the USDA Food Composition Databases
url = 'https://api.nal.usda.gov/fdc/v1/foods/search'
# API key (replace with your own if you have one)
api_key = 'IphEUj1GUJWBEjPhPJRENqRokVbVTtAIoaMcXqdK'
# Load the model
model = load_model("keras_Model1.h5", compile=False)
# Load the labels
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------recommendation-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def recom(condition,message):
    # Load the Excel workbook
    workbook = openpyxl.load_workbook('data.xlsx')
    # Get the active sheet
    sheet = workbook.active
    # Define lists for the first and second columns
    column1_values = []
    column2_values = []

    # Loop through each row in the sheet and append the first and second column values to the lists
    for row in sheet.iter_rows(min_row=2, values_only=True):
        column1_values.append(row[0])
        column2_values.append(row[1])
    for i  in range(len(column1_values)):
        if condition in column1_values[i]:
            print(column2_values[i])
            arturl=column2_values[i]
            
            bot.send_message(message.chat.id, text="check out this article, it may help you ", parse_mode='HTML')
            bot.send_message(message.chat.id, text=arturl, parse_mode='HTML')   
            
            url = f"https://www.amazon.in/s?k=product for {condition}"
            link_text = 'Check out this website!'
            message_text = f'<a href="{url}">{link_text}</a>'
            bot.send_message(message.chat.id, text=message_text, parse_mode='HTML')        
            bot.send_photo(message.chat.id, photo=url)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------conditions-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
user_data = {}
@bot.message_handler(commands=['conditions'])
def ask_conditions(message):
    keyboard = InlineKeyboardMarkup()
    for condition in conditions:
        keyboard.add(InlineKeyboardButton(text=condition, callback_data=condition))
    bot.send_message(chat_id=message.chat.id, text='Please select any conditions that apply to you:', reply_markup=keyboard)
    # Add user data to dictionary
    user_data[message.chat.id] = {'conditions': {}, 'submitted': False}
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    if call.data == 'submit':
        # Save the selected conditions to the user_data dictionary
        selected_conditions = [condition for condition, selected in user_data[call.message.chat.id]['conditions'].items() if selected]
        user_data[call.message.chat.id]['conditions'] = selected_conditions
        user_data[call.message.chat.id]['submitted'] = True
        bot.answer_callback_query(callback_query_id=call.id, text=f'Selected conditions: {selected_conditions}')
    else:
        condition = call.data
        user_data[call.message.chat.id]['conditions'][condition] = not user_data[call.message.chat.id]['conditions'].get(condition, False)
        button_text = f'{condition} ✅' if user_data[call.message.chat.id]['conditions'][condition] else condition
        keyboard = InlineKeyboardMarkup()
        for condition, selected in user_data[call.message.chat.id]['conditions'].items():
            button_text = f'{condition} ✅' if selected else condition
            keyboard.add(InlineKeyboardButton(text=button_text, callback_data=condition))
        bot.edit_message_reply_markup(chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=keyboard)

        bot.send_message(chat_id=call.message.chat.id, text='You have submitted your medical condition')       
        cnx = mysql.connector.connect(user='root', password='mes20ad048', host='127.0.0.1', database='pythondatas')
        cursor = cnx.cursor()
        sql = "INSERT INTO userconditions (uid,conditions) VALUES (%s, %s)"
        val = (call.message.chat.id,condition)
        cursor.execute(sql, val)      
        cnx.commit()
        cursor.close()
        cnx.close()
        print(user_data)
        # Schedule the function to be called every minute
        #recom(condition,call.message)
    #schedule.every(1).minutes.do(recom(condition,call.message))
        # Keep running the scheduler
    #while True:
     #   schedule.run_pending()
        #time.sleep(1)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------/book---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Define the layout of the form
markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
item1 = types.KeyboardButton('Send Name')
markup.add(item1)
# Ask the user to enter their details
@bot.message_handler(commands=['book'])
def send_welcome(message):
    msg = bot.send_message(message.chat.id, "Please enter your name:")
    bot.register_next_step_handler(msg, process_name_step)
def process_name_step(message):
    try:
        chat_id = message.chat.id
        name = message.text
        # Ask for the user's age
        msg = bot.send_message(chat_id, "Please enter your age:")
        bot.register_next_step_handler(msg, lambda age_msg: process_age_step(age_msg, chat_id, name))
    except Exception as e:
        bot.send_message(chat_id, "Oops, something went wrong. Please try again later.")
def process_age_step(message, chat_id, name):
    try:
        age = int(message.text)
        # Ask for the user's phone number
        msg = bot.send_message(chat_id, "Please enter your phone number:")
        bot.register_next_step_handler(msg, lambda phone_msg: process_phone_step(phone_msg, chat_id, name, age))
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid age (a number).")
        msg = bot.send_message(chat_id, "Please enter your age:")
        bot.register_next_step_handler(msg, lambda age_msg: process_age_step(age_msg, chat_id, name))
def process_phone_step(message, chat_id, name, age):
    try:
        phone_number = message.text
        # Ask for the appointment date
        msg = bot.send_message(chat_id, "Please enter the date for your appointment (YYYY-MM-DD):")
        bot.register_next_step_handler(msg, lambda date_msg: process_date_step(date_msg, chat_id, name, age, phone_number))
    except Exception as e:
        bot.send_message(chat_id, "Oops, something went wrong. Please try again later.")
def process_date_step(message, chat_id, name, age, phone_number):
    try:
        date = message.text
        # Ask for the appointment time
        msg = bot.send_message(chat_id, "Please enter the time for your appointment (HH:MM):")
        bot.register_next_step_handler(msg, lambda time_msg: process_time_step(time_msg, chat_id, name, age, phone_number, date))
    except Exception as e:
        bot.send_message(chat_id, "Oops, something went wrong. Please try again later.")
def process_time_step(message, chat_id, name, age, phone_number, date):
    try:
        time = message.text
        # Ask for the reason for the appointment
        msg = bot.send_message(chat_id, "Please enter the reason for your appointment:")
        bot.register_next_step_handler(msg, lambda reason_msg: process_reason_step(reason_msg, chat_id, name, age, phone_number, date, time))
    except Exception as e:
        bot.send_message(chat_id, "Oops, something went wrong. Please try again later.")
def process_reason_step(message, chat_id, name, age, phone_number, date, time):
    try:
        reason = message.text
        cnx = mysql.connector.connect(user='root', password='mes20ad048', host='127.0.0.1',database='pythondatas')
        cursor = cnx.cursor()
        insert_query = "INSERT INTO appointments (name, age, phone_number, date, time, reason) VALUES (%s, %s, %s, %s, %s, %s)"
        data = (name, age, phone_number, date, time, reason)
        cursor.execute(insert_query, data)
        cnx.commit()
        cursor.close()
        cnx.close()
        # Send confirmation message to user
        confirmation_message = f"Thank you, {name}. Your appointment has been scheduled for {date} at {time}. We will contact you at {phone_number} if we need to reschedule or cancel. Thank you for choosing our service."
        bot.send_message(chat_id, confirmation_message)
    except Exception as e:
        bot.send_message(chat_id, "Oops, something went wrong. Please try again later.")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
@bot.message_handler(commands=['reminder'])
def handle_reminder_command(message):
    token_bot2 = '6254586187:AAGOdBfyQyk6UMoowW494xuOXM2VYrldkF4'
    bot2 = telebot.TeleBot(token_bot2)
    # Extract the chat ID from the incoming message
    chat_id = message.chat.id
    
    # Create a deep-link to the second bot
    bot2_deep_link = f'https://t.me/{bot2.get_me().username}'
    
    # Generate the redirection message
    redirection_message = f'Click [here]({bot2_deep_link}) to set reminder.'
    
    # Send the redirection message as a reply
    bot.send_message(chat_id, redirection_message, parse_mode='Markdown')

#---------------------------------------------------chat----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
@bot.message_handler(content_types=['text'])

def handle_response(message):
    # Create your own response logic
    phrases = ['how are you', 'what is your name', 'what can you do', 'who created you']
    processed = message.text.lower()
    print(processed)
    # Set up your OpenAI API credentials
    openai.api_key = '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'

    # Function to interact with the GPT-3 model
    def query_gpt(question):
        response = openai.Completion.create(
            engine='davinci',
            prompt=question,
            max_tokens=100,
            n=1,
            stop=None,
        )

        return response.choices[0].text.strip()
    
    if 'i have ' in processed:
        try: 
            inpt=message.text.lower()
            ert=inpt.replace("i have","")
            user_symptoms = [s.strip() for s in ert.split(',')]
            bot.send_message(message.chat.id, "Analysing...")
            
            symptoms = []
            diseases = []
            for symptom, associated_diseases in symptoms_to_diseases.items():
                 symptoms.append(symptom)
                 diseases.append(', '.join(associated_diseases))
            # Vectorize the symptoms
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(symptoms)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(diseases)
            classifier = RandomForestClassifier()
            classifier.fit(X, y)
            X_user = vectorizer.transform(user_symptoms)
            predicted_diseases_encoded = classifier.predict(X_user)    
            predicted_diseases = label_encoder.inverse_transform(predicted_diseases_encoded)
            if len(predicted_diseases) > 0:
                bot.send_message(message.chat.id, "Predicted diseases based on the given symptoms:")
                bot.send_message(message.chat.id,predicted_diseases)
            else:
                print("No diseases predicted for the given symptoms.")
                bot.send_message(message.chat.id, " Please keep in mind that the predicted diseases may vary. My ability to predict diseases relies solely on the data provided to me.")
        except ValueError:
                bot.send_message(message.chat.id, "Please enter  valid symptoms.")

    elif 'hi' in processed:
        bot.send_message(message.chat.id, 'Hi there! 👋')   
    elif 'what is your name' in processed:
        bot.send_message(message.chat.id, 'my name is medbot.ai !') 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#                
        
    elif '/help' in processed:
        bot.send_message(message.chat.id, 'Welcome to the Help section for the MedBot!\n\nThis bot is designed to help you with various health-related tasks and provide information on different medical topics. Here\'s what you can do with this bot:\n\n• /cam ---for image recognition \n• /book ---for booking appointments\n• /notify ---for reminders\n\n -- symptoms fever,cough,etc')
    
    elif '/start' in processed:
        bot.send_message(message.chat.id, 'Welcome to the Help section for the MedBot!')

        

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    elif 'how are you' in processed:
        bot.send_message(message.chat.id, 'I\'m good! ')
        bot.send_message(message.chat.id, 'thanks for asking 😊 ')
    elif 'hello' in processed:
        bot.send_message(message.chat.id, 'hi there')
    elif 'who are you' in processed:
        bot.send_message(message.chat.id, 'i am medbot')
    elif 'what can you do' in processed:
        bot.send_message(message.chat.id, 'I can provide information, answer questions, and have conversations')
    elif 'who created you' in processed:
        bot.send_message(message.chat.id, "I was created by a team of developers at MESCE \nFor more details go to ")
    elif "find what" in processed:
        print(processed)
        
                # Search the query in Wikipedia
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        try:
            bot.send_message(message.chat.id, "Searching...")
            summary = wikipedia.summary(processed)
            bot.send_message(message.chat.id, summary)
        except wikipedia.exceptions.DisambiguationError as e:
            # If the query is ambiguous, choose the first option
            summary = wikipedia.summary(e.options[0])
            bot.send_message(message.chat.id, summary)
        except wikipedia.exceptions.PageError:
            # If no Wikipedia page was found for the query
            bot.send_message(message.chat.id, 'Sorry, I could not find any information on that topic.')
        except requests.exceptions.ConnectionError:
            # If there was an error connecting to Wikipedia
            bot.send_message(message.chat.id, 'Sorry, there was an error connecting to the world . Please try again later.')
    else :
        try:
            # Function to send a loading effect
            def send_loading_effect(chat_id, duration=3):
                message = bot.send_message(chat_id, 'Loading...')
                end_time = time.time() + duration
                while time.time() < end_time:
                    bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text='.')
                    time.sleep(1)
                    bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text='..')
                    time.sleep(1)
                    bot.edit_message_text(chat_id=chat_id, message_id=message.message_id, text='...')
                    time.sleep(1)
                bot.delete_message(chat_id, message.message_id)
            chat_id=message.chat.id
            qry=processed
            response = ask_gpt(qry+"  is this query is related to  helth or medical field or life,comman queries only say 'yes' or 'no'")
            send_loading_effect(chat_id)
            # Use GPT-3 for other queries
            
            print('Bot:', response)
            rsp=response.lower()
            print(rsp)
            if "yes" in rsp:
                rslt=ask_gpt(qry)
                bot.send_message(message.chat.id,  rslt)
            elif "no"in rsp:
                bot.send_message(message.chat.id, 'Sorry, i dont understand your message')
                bot.send_message(message.chat.id,  " i can only response to helth related queries ")
        except:
            bot.send_message(message.chat.id, 'Sorry, i dont understand your message')
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------image-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        print("loading...")
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open('received_image.jpg', 'wb') as new_file:
            new_file.write(downloaded_file)
            print("image downloaded..")

        # Add some delay to simulate image processing time
        time.sleep(2)

        image = Image.open("received_image.jpg").convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the normalized image into the data array
        # Please ensure that the 'data' array is properly defined before using this code.
        data[0] = normalized_image_array

        # Predict the model
        # Please ensure that the 'model' is correctly defined and loaded before using this code.
        prediction = model.predict(data)

        # Get the index of the predicted class
        index = np.argmax(prediction)

        # Please ensure that the 'class_names' list is defined and contains class labels.
        class_name = class_names[index]

        # Send the class name as a message to the user
        bot.send_message(message.chat.id, "Name: " + class_name[2:])

        food_name = class_name[2:]
        print(food_name)
        if "none edible item" in food_name:
            nutrient_info ="Could not find"
            if "Could not find" in nutrient_info:
                bot.send_message(message.chat.id, "Could not find any information about it,i think it is not edible ")
        else:
            nutrient_info = get_nutrient_info(food_name)
            if "Could not find" in nutrient_info:
                bot.send_message(message.chat.id, "Could not find any information about it, maybe my team forgot to teach me.")
            else:
                bot.send_message(message.chat.id, nutrient_info)
    except Exception as e:
        # If any error occurs during image processing or other steps, send a random error message to the user.
        print("Error:", e)
        error_msg = random.choice(error_messages)
        bot.send_message(message.chat.id, error_msg)
     
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#





#------------------------------------------------------------------------------runbot-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#  
bot.polling(none_stop=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

