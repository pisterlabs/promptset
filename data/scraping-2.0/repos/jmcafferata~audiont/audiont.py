# -*- coding: utf-8 -*-
# This code is used to process audio files from the telegram bot, get the text from the audio file and generate an automated response.
# Este código se usa para procesar archivos de audio del bot de telegram, obtener el texto del archivo de audio y generar una respuesta automatizada.

# import the necessary libraries // importar las librerías necesarias
# library used to define the types of variables // librería usada para definir los tipos de variables
from typing import DefaultDict, Optional, Set
# library used to handle dictionaries // librería usada para manejar diccionarios
from collections import defaultdict
from telegram.ext import (
    Application,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ExtBot,
    TypeHandler,
    InlineQueryHandler,
    MessageHandler,
    filters
)
# library used to handle conversations // librería usada para manejar conversaciones
from telegram.ext import ConversationHandler
# import the clean_options file // importar el archivo clean_options
import modules.clean_options as clean
import re  # library used to handle regular expressions // librería usada para manejar expresiones regulares
# library used to handle dates // librería usada para manejar fechas
from datetime import datetime
import csv  # library used to handle csv files // librería usada para manejar archivos csv
import config as config  # import the config file // importar el archivo de configuración
# library used to communicate with the Telegram bot // librería usada para comunicarse con el bot de Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
# import the decode_utf8 file // importar el archivo decode_utf8
import modules.decode_utf8
# import the convert_to_wav file // importar el archivo convert_to_wav
import modules.convert_to_wav
# import the store_to_csv file // importar el archivo store_to_csv
import modules.csv_manipulation as csvm
# import the ai_functions file // importar el archivo ai_functions
import modules.ai_functions as ai
# import the subscriptions file // importar el archivo subscriptions
import modules.subscriptions as subs
import pathlib as Path
import requests
import pytz
import urllib.request
import telegram.constants
import logging
import requests
from bs4 import BeautifulSoup
import openai
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import traceback
import json
from telegram.constants import ParseMode

# get the instructions from json file data.json



def get_instructions():
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    key = data['instructions']
    instructions = read_data_from_csv(key=key, filename="instructions.csv")
    personality = read_data_from_csv(key="personalidad", filename="instructions.csv")
    return instructions, personality, key

def escape_markdown_v2(text):
    # List of special characters that need to be escaped in Markdown V2
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    # Escape each special character with a preceding backslash
    escaped_text = ''.join(['\\' + char if char in special_chars else char for char in text])
    return escaped_text


# read key-value pairs from csv file // leer pares clave-valor de un archivo csv
def read_data_from_csv(key: str, filename: str) -> dict:
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row[0].strip() == key.strip():
                # return the value of the key // devolver el valor de la clave
                return row[1]
    # return None if the key is not found // devolver None si la clave no se encuentra
    return None

# replace key-value pairs in csv file // reemplazar pares clave-valor en un archivo csv
def update_data_file(key: str, value: str, filename: str) -> None:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data[key] = value

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# CONTANTS // CONSTANTES

# define the states of the conversation // definir los estados de la conversación
ASK_NAME, ASK_DESCRIPTION,ASK_MESSAGES,AWAIT_INSTRUCTIONS,ASK_MORE_MESSAGES,ASK_PAYEE,CONFIRM_PAYMENT = range(7)

# define the bot instructions

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    
    await update.message.chat.send_chat_action(action=telegram.constants.ChatAction.TYPING)

    try:

        response = None # Initialize the 'response' variable here
        
        instructions, personality, key = get_instructions()
    
        # get sender username
        username = update.message.from_user.username
        #get sender full name
        full_name = update.message.from_user.full_name

        #check if username aint empty
        if username == None:
            username = full_name

        #if sender isn't owner
        if username != my_username:
            response = await ai.secretary(update,update.message.text, personality,context)
           

        else:
            if key == "chat3":
                # call the chat function // llamar a la función chat
                response = await ai.chat(update,update.message.text,"3", personality)
            elif key == "chat4":
                # call the chat function // llamar a la función chat
                response = await ai.chat(update,update.message.text,"4",personality)
            elif key == "crud":

                response = await ai.crud(update, update.message.text,context)
            await update.message.reply_text(response)
    
        return
    except Exception as e:
        # print and send the formatted traceback // imprimir y enviar el traceback formateado
        traceback = sys.exc_info()[2]
        traceback.print_exc()
        await update.message.reply_text(traceback.format_exc())
        

# function that handles the voice notes // función principal que maneja las notas de voz
async def handle_voice(update, context):
    
    await update.message.chat.send_chat_action(action=telegram.constants.ChatAction.TYPING)

    # call the transcribe_audio function // llamar a la función transcribe_audio
    try:
        transcription = await ai.transcribe_audio(update)

        response = None # Initialize the 'response' variable here
        
        instructions, personality, key = get_instructions()
    
        # get sender username
        username = update.message.from_user.username
        #get sender full name
        full_name = update.message.from_user.full_name

        #check if username aint empty
        if username == None:
            username = full_name


         #if sender isn't jmcafferata
        if username != my_username:
            response = await ai.secretary(update,transcription, personality,context)
            await update.message.reply_text(response)

        else:
            if key == "chat3":
                # call the chat function // llamar a la función chat
                response = await ai.chat(update,transcription,"3",personality)
                await update.message.reply_text(response)
            elif key == "chat4":
                # call the chat function // llamar a la función chat
                response = await ai.chat(update,transcription,"4",personality)
                await update.message.reply_text(response)
            elif key == "crud":

                await ai.crud(update, transcription,context)
        
        print('################# SENDING MESSAGE #################')
        
        # if the response is an array, send each element of the array as a message // si la respuesta es un array, enviar cada elemento del array como un mensaje
        
        return
    except Exception as e:
        # assign the traceback to a variable // asignar el traceback a una variable
        traceback = sys.exc_info()[2]
        traceback = traceback[:79]
        # print and send the formatted traceback // imprimir y enviar el traceback formateado
        traceback.print_exc()
        await update.message.reply_text(traceback.format_exc())
        
    return ConversationHandler.END

async def handle_audio(update, context):

    #get username and name
    username = update.message.from_user.username
    full_name = update.message.from_user.full_name
    #check if username is none  
    if username == None:
        username = full_name
    

    await update.message.chat.send_chat_action(action=telegram.constants.ChatAction.TYPING)

    instructions, personality, key = get_instructions()
    # call the transcribe_audio function // llamar a la función transcribe_audio
        
    try:
        transcription = await ai.transcribe_audio(update)
        # if there's an error send a message // si hay un error enviar un mensaje
    except:
        pass
    else:
        if username != my_username:
            #send transcription to user and end
            await update.message.reply_text("El audio dice:")
            await update.message.reply_text(transcription)
            return

        csvm.store_to_csv(message=transcription)
        # reply to the message with the text extracted from the audio file // responder al mensaje con el texto extraído del archivo de audio
        await update.message.reply_text("El audio dice:")
        await update.message.reply_text(transcription)
        
        # call the prompt function // llamar a la función prompt
        response = await ai.complete_prompt(reason="summary", message=transcription, username=update.message.from_user.username,update=update,personality=personality)
        # call the clean_options function // llamar a la función clean_options
        response_text, options = await clean.clean_options(response)
        # add the options to the current response options
        for option in options:
            current_response_options.append(option)
        # reply to the message with the summary and the 5 options // responder al mensaje con el resumen y las 5 opciones
        await update.message.reply_text(response_text, reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(text=options[0], callback_data="0")
                ],
                [
                    InlineKeyboardButton(text=options[1], callback_data="1")
                ],
                [
                    InlineKeyboardButton(text=options[2], callback_data="2")
                ],
                [
                    InlineKeyboardButton(text=options[3], callback_data="3")
                ]
            ]
        ))
        return AWAIT_INSTRUCTIONS            
 

# function that handles the voice notes when responding to a voice note // función principal que maneja las notas de voz cuando se responde a una nota de voz
async def respond_audio(update, context):
    instructions, personality, key = get_instructions()
    # call the transcribe_audio function // llamar a la función transcribe_audio
    transcription = await ai.transcribe_audio(update)
    response = await ai.complete_prompt(reason="answer", message=transcription, username=update.message.from_user.username,update=update,personality=personality)
    await update.message.reply_text(response)
    return ConversationHandler.END
   
# handles when the bot receives something that is not a command // maneja cuando el bot recibe algo que no es un comando
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text="No entiendo ese comando.")


# Function to handle files
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Extract the file and its description from the message
    file = await update.message.document.get_file()
    description = update.message.caption
    #print the file components
    

    # Get the user's username
    username = update.message.from_user.username

  # Save the file to the users/{username}/files folder
    file_path = "users/"+username+"/files/"
    print('file id: '+file.file_id)
    print('file path: '+file.file_path)
    print('file uid: '+file.file_unique_id)
    print('file size: '+str(file.file_size))

    # Remove the duplicate file path assignment
    new_file_path = file_path + file.file_path.split('/')[-1]
    print('new file path: '+new_file_path)

    # Make sure that the directories in the file path exist
    import os
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    urllib.request.urlretrieve(file.file_path, new_file_path)


    timezone = pytz.timezone("America/Argentina/Buenos_Aires")
    # Store the file path, the time of the message (in Buenos Aires time), and the description
    now = datetime.now(timezone)
    file_entry = f"{now.strftime('%d/%m/%Y %H:%M:%S')}|{description}: {new_file_path}|{ai.get_embedding(description,'text-embedding-ada-002')}\n"
  
    # Save the entry to the users/{username}/messages.csv file
    with open(f"users/{username}/messages.csv", "a", encoding="utf-8") as f:
        f.write(file_entry)

    # Send a confirmation message to the user
    await update.message.reply_text("Archivo y descripción guardados correctamente.")

async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):

    instructions, personality, key = get_instructions()
    # get the data from the callback query // obtener los datos de la consulta de devolución de llamada
    data = update.callback_query.data
    print("Data: "+data)
    # if the data is a number, then it's an instruction // si los datos son un número, entonces es una instrucción
    if data.isdigit():
        print("current_response_options: "+str(current_response_options))
        response = await ai.complete_prompt("answer", current_response_options[int(data)], update.callback_query.from_user.username, update, personality)
        # send a message saying that if they didn't like the response, they can send a voice note with instructions // enviar un mensaje diciendo que si no les gustó la respuesta, pueden enviar una nota de voz con instrucciones
        await update.callback_query.message.reply_text(response)
        await update.callback_query.message.reply_text("🥲 Si no te gustó la respuesta, podés mandarme una nota de voz con instrucciones 🗣️ o apretar otro botón.")
        return AWAIT_INSTRUCTIONS
    
async def google(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Data: ['explain', 'green', 'hydrogen', 'news', 'in', 'a', 'few', 'steps'] make them a string
    data = ''
    for arg in context.args:
        data = data + arg + " "
    print("Doing google search...")
    print("Data: "+data)
    # if the data is a number, then it's an instruction // si los datos son un número, entonces es una instrucción
    try:
        response = await ai.complete_prompt("google", data, None,update)
        await update.message.reply_text(response)
    except Exception as e:
        exception_traceback = traceback.format_exc()
        await update.message.reply_text(f"¡No encontré nada! 🙃\nException: {str(e)}\nTraceback: {exception_traceback}")
        print('⬇️⬇️⬇️⬇️ Error en google ⬇️⬇️⬇️⬇️\n',exception_traceback)

    
    # send a message saying that if they didn't like the response, they can send a voice note with instructions // enviar un mensaje diciendo que si no les gustó la respuesta, pueden enviar una nota de voz con instrucciones
    
async def modo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Data: ['explain', 'green', 'hydrogen', 'news', 'in', 'a', 'few', 'steps'] make them a string
    data = ''
    for arg in context.args:
        data = data + arg + " "
    data = data.strip()
    print("Data: "+data)
    # if the data is a number, then it's an instruction // si los datos son un número, entonces es una instrucción
    try:
        instructions = read_data_from_csv(key=data,filename="instructions.csv")
        print("Instructions: "+instructions)
        update_data_file(key="instructions",value=data,filename="data.json")
        print("Instructions set")
        await update.message.reply_text("¡Listo! Ahora tengo una nueva personalidad:\n "+instructions)
        
    except Exception as e:
        exception_traceback = traceback.format_exc()
        print('⬇️⬇️⬇️⬇️ Error en instructions ⬇️⬇️⬇️⬇️\n',exception_traceback)

async def chat3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Data: ['explain', 'green', 'hydrogen', 'news', 'in', 'a', 'few', 'steps'] make them a string
    try:
        instructions = read_data_from_csv(key="chat3",filename="instructions.csv")
        print("Instructions: "+instructions)
        update_data_file(key="instructions",value="chat3",filename="data.json")
        print("Instructions set")
        await update.message.reply_text("¡Listo! Ahora tengo una nueva personalidad:\n "+instructions)
        
    except Exception as e:
        exception_traceback = traceback.format_exc()
        print('⬇️⬇️⬇️⬇️ Error en instructions ⬇️⬇️⬇️⬇️\n',exception_traceback)

async def chat4(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Data: ['explain', 'green', 'hydrogen', 'news', 'in', 'a', 'few', 'steps'] make them a string
    try:
        instructions = read_data_from_csv(key="chat4",filename="instructions.csv")
        print("Instructions: "+instructions)
        update_data_file(key="instructions",value="chat4",filename="data.json")
        print("Instructions set")
        await update.message.reply_text("¡Listo! Ahora tengo una nueva personalidad:\n "+instructions)
        
    except Exception as e:
        exception_traceback = traceback.format_exc()
        print('⬇️⬇️⬇️⬇️ Error en instructions ⬇️⬇️⬇️⬇️\n',exception_traceback)

async def crud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Data: ['explain', 'green', 'hydrogen', 'news', 'in', 'a', 'few', 'steps'] make them a string
    try:
        instructions = read_data_from_csv(key="crud",filename="instructions.csv")
        print("Instructions: "+instructions)
        update_data_file(key="instructions",value="crud",filename="data.json")
        print("Instructions set")
        await update.message.reply_text("¡Listo! Ahora tengo una nueva personalidad:\n "+instructions)
        
    except Exception as e:
        exception_traceback = traceback.format_exc()
        print('⬇️⬇️⬇️⬇️ Error en instructions ⬇️⬇️⬇️⬇️\n',exception_traceback)
 

# main function // función principal
if __name__ == '__main__':

    current_response_options = []

    my_username = config.my_username

    # create the bot // crear el bot
    application = Application.builder().token(config.telegram_api_key).build()

    # use ai.generate_embeddings(file,column) to turn the column full_text in users/jmcafferata/justTweets.csv into a list of embeddings
    # print(ai.generate_embeddings("users/jmcafferata/cleandataNoRows.csv","full_text"))

 

    # for when the bot receives an audio file // para cuando el bot recibe un archivo de audio
    audio_handler = MessageHandler(filters.AUDIO, handle_audio)
    application.add_handler(audio_handler)

    
    # for when the bot receives a text message // para cuando el bot recibe un archivo de audio
    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text)
    application.add_handler(text_handler)


    # for when the bot receives a voice note // para cuando el bot recibe una nota de voz
    # exclude conversation states // excluir estados de conversación
    voice_handler = MessageHandler(filters.VOICE & (~filters.COMMAND), handle_voice)
    application.add_handler(voice_handler)

    file_handler = MessageHandler(filters.Document.ALL, handle_file)
    application.add_handler(file_handler)

    #/google command
    google_handler = CommandHandler('google', google)
    application.add_handler(google_handler)

    #/google command
    modo_handler = CommandHandler('modo', modo)
    application.add_handler(modo_handler)

    #/chat3 command
    chat3_handler = CommandHandler('chat3', chat3)
    application.add_handler(chat3_handler)

    #/chat4 command
    chat4_handler = CommandHandler('chat4', chat4)
    application.add_handler(chat4_handler)

    crud_handler = CommandHandler('crud', crud)
    application.add_handler(crud_handler)


    # a callback query handler // un manejador de consulta de devolución de llamada
    callback_handler = CallbackQueryHandler(callback=callback)
    application.add_handler(callback_handler)

    # start the bot // iniciar el bot
    application.run_polling()
    # logger.info('Bot started')