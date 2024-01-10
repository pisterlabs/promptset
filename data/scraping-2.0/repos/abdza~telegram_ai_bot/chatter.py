#!/usr/bin/env python

import telebot
from openai import OpenAI
import settings
import os
import textract
import pprint
import base64
# import chromadb
# import threading
# import tiktoken
# from re import template
# from urlextract import URLExtract
# import urllib
# from chromadb.config import Settings
import time
from uuid import uuid4
import sqlite3
from datetime import datetime, timedelta
import yahooquery as yq
import numpy as np
from numerize import numerize
from sklearn.cluster import KMeans
from pydub import AudioSegment
import pandas as pd
from tabulate import tabulate

bot = telebot.TeleBot(settings.telebot_key)
# openai.api_key = settings.openai_key

script_path = os.path.abspath(__file__)

# Get the directory containing the current script
script_dir = os.path.dirname(script_path)

chat_model = "gpt-4-1106-preview"
# chat_model = "gpt-4"
# chat_model = "gpt-3.5-turbo"

pp = pprint.PrettyPrinter(indent=4)


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    finalfilepath = os.path.join(script_dir,filepath)
    with open(finalfilepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

AIClient = OpenAI(
    api_key=settings.openai_key
)
if settings.AIAssistantID:
    AIAssistant = AIClient.beta.assistants.update(settings.AIAssistantID,instructions=open_file('system_ai_friend.txt'))
else:
    AIAssistant = AIClient.beta.assistants.create(
        name="Telegram Bot",
        instructions=open_file('system_ai_friend.txt'),
        tools=[{"type": "code_interpreter"},{"type": "retrieval"}],
        model="gpt-4-1106-preview"
    )

def update_db():
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS threads (id INTEGER PRIMARY KEY, timestamp, thread_id TEXT, user_id TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS subscribers (id INTEGER PRIMARY KEY, user_id TEXT, service TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS chat (id INTEGER PRIMARY KEY, timestamp, role TEXT, user TEXT, chat TEXT, message TEXT, tokens INTEGER, costs REAL)")
    con.commit()
    con.close()

def append_message(message,content,message_role='assistant'):
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()

    thread = cursor.execute("SELECT * FROM threads where user_id = ? ORDER BY timestamp desc limit 1",(message.from_user.id,)).fetchall()
    ai_thread = None
    print("Threads: ", thread)
    if not thread:
        ai_thread = AIClient.beta.threads.create()
        cursor.execute("INSERT INTO threads (timestamp, thread_id, user_id) VALUES (datetime('now'), ?, ?)", (ai_thread.id, message.from_user.id))
    else:
        ai_thread = AIClient.beta.threads.retrieve(thread[0][2])

    print("AI Thread:",ai_thread)
    print("User Id:", message.from_user.id)
    print("Chat Id:", message.chat.id)

    if ai_thread:
        AIClient.beta.threads.messages.create(ai_thread.id,role=message_role,content=content)

    con.commit()
    con.close()

def get_response(message,content):
    response = "Hi there!"
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()
    subscription = cursor.execute("select * from subscribers where user_id = ? and service = ?",(message.from_user.id,'AICHAT')).fetchall()
    if len(subscription)>0:

        thread = cursor.execute("SELECT * FROM threads where user_id = ? ORDER BY timestamp desc limit 1",(message.from_user.id,)).fetchall()
        ai_thread = None
        print("Threads: ", thread)
        if not thread:
            ai_thread = AIClient.beta.threads.create()
            cursor.execute("INSERT INTO threads (timestamp, thread_id, user_id) VALUES (datetime('now'), ?, ?)", (ai_thread.id, message.from_user.id))
        else:
            ai_thread = AIClient.beta.threads.retrieve(thread[0][2])

        print("AI Thread:",ai_thread)
        print("User Id:", message.from_user.id)
        print("Chat Id:", message.chat.id)

        if ai_thread:
            send_message = AIClient.beta.threads.messages.create(ai_thread.id,role='user',content=content)
            print("Sent message:",content)

            message_run = AIClient.beta.threads.runs.create(
                thread_id=ai_thread.id,
                assistant_id=AIAssistant.id
            )

            while message_run.status !="completed":
                message_run = AIClient.beta.threads.runs.retrieve(
                    thread_id=ai_thread.id,
                    run_id=message_run.id
                )
                print(message_run.status)

            messages = AIClient.beta.threads.messages.list(
                thread_id=ai_thread.id
            )

            print(messages.data[0].content[0].text.value)
            response = messages.data[0].content[0].text.value

        con.commit()
        con.close()

    return response

@bot.message_handler(commands=['reset'])
def reset(message):
    try:
        con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
        cursor = con.cursor()
        cursor.execute("delete from chat where user = ?",(message.from_user.id,))
        con.commit()
        con.close()
        response = get_response(message,'Hello. My name is ' + message.from_user.first_name)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['subscribe'])
def subscribe(message):
    try:
        tokens = message.text.split(' ')
        if len(tokens)<2:
            bot.reply_to(message, "Please specify the service you'd like to subscribe to")
        else:
            service = tokens[1].upper()
            con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
            cursor = con.cursor()
            subscription = cursor.execute("select * from subscribers where user_id = ? and service = ?",(message.from_user.id,service)).fetchall()
            if len(subscription)>0:
                bot.reply_to(message, "You are already subscribed to " + service + " subscription")
            else:
                failed = False
                if service=='AICHAT':
                    if len(tokens)<3:
                        bot.reply_to(message, "Your subscription to " + service + " failed")
                        failed = True
                    else:
                        if tokens[2]!=settings.chat_password:
                            bot.reply_to(message, "Your subscription to " + service + " failed")
                            failed = True
                if not failed:
                    cursor.execute("INSERT INTO subscribers (user_id, service) VALUES (?, ?)", (message.from_user.id, service))
                    con.commit()
                    con.close()
                    bot.reply_to(message, "You have succesfully subscribed to " + service + " subscription")
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['unsubscribe'])
def unsubscribe(message):
    try:
        tokens = message.text.split(' ')
        if len(tokens)<2:
            bot.reply_to(message, "Please specify the service you'd like to unsubscribe to")
        else:
            service = tokens[1].upper()
            con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
            cursor = con.cursor()
            subscription = cursor.execute("select * from subscribers where user_id = ? and service = ?",(message.from_user.id,service)).fetchall()
            if len(subscription)>0:
                cursor.execute("delete from subscribers where user_id = ? and service = ?",(message.from_user.id,service))
                con.commit()
                con.close()
                bot.reply_to(message, "You have been unsubscribed from " + service + " subscription")
            else:
                bot.reply_to(message, "You are not subscribed to " + service + " subscription")
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['length','size'])
def msg_length(message):
    try:
        con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
        cursor = con.cursor()
        messages = cursor.execute("SELECT count(*) as msglength,sum(tokens) as tokensum, sum(costs) as cost_total FROM chat WHERE chat = ?", (message.chat.id,)).fetchall()
        response = 'Message Id:' + str(message.chat.id) + ' Msg count: ' + str(messages[0][0]) + ' messages. Total tokens: ' + str(messages[0][1]) + ' . Total cost: ' + str(messages[0][2])
        con.close()
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['setup','start'])
def setup(message):
    try:
        response = get_response(message,'Hello. My name is ' + message.from_user.first_name)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['stock','ticker'])
def stock(message):
    try:
        tokens = message.text.split(' ')
        ticker = tokens[1].upper()
        yqticker = yq.Ticker(ticker)
        end_date = datetime.now()
        days = 120
        start_date = end_date - timedelta(days=days)
        candles = yqticker.history(start=start_date,end=end_date,interval='1d')
        
        response = get_response(message,"Ticker " + ticker + " candles: " + str(candles))
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['results'])
def results(message):
    try:
        results = pd.read_csv(os.path.join(script_dir,'results.csv'))
        tosend = results[['ticker','marks','price']]
        bot.reply_to(message, tabulate(tosend.iloc[-5:],headers="keys"))
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['props'])
def stock_props(message):
    try:
        results = pd.read_csv(os.path.join(script_dir,'results.csv'))
        tokens = message.text.split(' ')
        if len(tokens)>1:
            for tk in tokens[1:]:
                fresult = results[results['ticker']==tk.upper()]
                if fresult.empty:
                    bot.reply_to(message,'Info for ticker ' + tk + ' could not be found')
                else:
                    response = fresult['prop']
                    bot.reply_to(message, response)
                    response = fresult['levels']
                    bot.reply_to(message, response)
        else:
            bot.reply_to(message,"Need to provide ticker to see")
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(commands=['levels'])
def stock_levels(message):
    #try:
        tokens = message.text.split(' ')
        ticker = tokens[1].upper()
        yqticker = yq.Ticker(ticker)
        end_date = datetime.now()
        days = 200
        start_date = end_date - timedelta(days=days)
        candles = yqticker.history(start=start_date,end=end_date,interval='1d')
        minute_start_date = end_date - timedelta(days=1)
        minute_candles = yqticker.history(start=minute_start_date,end=end_date,interval='5m')

        response = "Levels:"
        min = candles['low'].min()
        max = candles['high'].max()
        p_range = candles['high'] - candles['low']
        range_avg = p_range.mean()
        vol_avg = candles['volume'].mean()
        min_vol_avg = minute_candles['volume'].mean()
        response += "\nStart: " + str(start_date)
        response += "\nEnd: " + str(end_date)
        response += "\nMin: " + str(min)
        response += "\nMax: " + str(max)
        response += "\nRange Avg: " + str(numerize.numerize(range_avg))
        response += "\nVol Avg: " + str(numerize.numerize(vol_avg))
        if min_vol_avg!=None and not np.isnan(min_vol_avg) and int(min_vol_avg)>100:
            response += "\n5 Min Vol Avg: " + str(numerize.numerize(min_vol_avg))

        datarange = max - min
        if datarange < 50:
            kint = int(datarange / 0.5)
        else:
            kint = int(datarange % 20)

        datalen = len(candles)

        highlevels = np.array(candles['high'])
        kmeans = KMeans(n_clusters=kint).fit(highlevels.reshape(-1,1))
        highclusters = kmeans.predict(highlevels.reshape(-1,1))

        resistancelevels = {}

        for cidx in range(datalen):
            curcluster = highclusters[cidx]
            if curcluster not in resistancelevels:
                resistancelevels[curcluster] = 1
            else:
                resistancelevels[curcluster] += 1

        donecluster = []
        finalreslevels = {}
        dresponse = ""
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = highclusters[cidx]
            if resistancelevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalreslevels[curcluster] = {'level':candle['high'],'count':1}
                else:
                    finalreslevels[curcluster] = {'level':(finalreslevels[curcluster]['level'] + candle['high'])/2,'count':finalreslevels[curcluster]['count']+1}

        response += "\n\nResistance levels:"
        for lvl,clstr in sorted(finalreslevels.items(),key=lambda x: x[1]['level']):
            response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])

        if datarange < 50:
            kint = int(datarange / 0.5)
        else:
            kint = int(datarange % 20)
        lowlevels = np.array(candles['low'])
        kmeans = KMeans(n_clusters=kint).fit(lowlevels.reshape(-1,1))
        lowclusters = kmeans.predict(lowlevels.reshape(-1,1))

        supportlevels = {}

        for cidx in range(datalen):
            curcluster = lowclusters[cidx]
            if curcluster not in supportlevels:
                supportlevels[curcluster] = 1
            else:
                supportlevels[curcluster] += 1

        donecluster = []
        finalsuplevels = {}
        dresponse = ""
        for cidx in range(datalen):
            candle = candles.iloc[cidx]
            curcluster = lowclusters[cidx]
            if supportlevels[curcluster] > 2:
                if curcluster not in donecluster:
                    donecluster.append(curcluster)
                    finalsuplevels[curcluster] = {'level':candle['low'],'count':1}
                else:
                    finalsuplevels[curcluster] = {'level':(finalsuplevels[curcluster]['level'] + candle['low'])/2,'count':finalsuplevels[curcluster]['count']+1}

        response += "\n\nSupport levels:"
        for lvl,clstr in sorted(finalsuplevels.items(),key=lambda x: x[1]['level']):
            response += "\n" + str(clstr['level']) + " : " + str(clstr['count'])
        
        response += "\n\n" + dresponse
        bot.reply_to(message, response)

@bot.message_handler(commands=['imagine'])
def imagine(message):
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()
    subscription = cursor.execute("select * from subscribers where user_id = ? and service = ?",(message.from_user.id,'AICHAT')).fetchall()
    if len(subscription)>0:
        print("Got image request:",message)
        try:
            response_image = AIClient.images.generate(
                model="dall-e-3",
                prompt=message,
                n=1,
                size="1024x1024"
            )
            image_url = response_image.data[0].url
            bot.send_photo(message.chat.id, image_url)
        except Exception as e:
            bot.reply_to(message, "Sorry, " + str(e))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@bot.message_handler(content_types=['photo'])
def photo_processing(message):
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()
    subscription = cursor.execute("select * from subscribers where user_id = ? and service = ?",(message.from_user.id,'AICHAT')).fetchall()
    if len(subscription)>0:
        try:
            file_info = bot.get_file(message.photo[0].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(os.path.join(script_dir,file_info.file_path), 'wb') as new_file:
                new_file.write(downloaded_file)
            response = AIClient.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message.caption},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + encode_image(os.path.join(script_dir,file_info.file_path)),
                            },
                        },
                    ],
                }
                ],
                max_tokens=300,
            )
            # print("Response:",response)
            bot.reply_to(message, response.choices[0].message.content)
        except Exception as e:
            bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(content_types=['document'])
def document_processing(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(os.path.join(script_dir,file_info.file_path), 'wb') as new_file:
            new_file.write(downloaded_file)
        filetext = textract.process(os.path.join(script_dir,file_info.file_path))
        usermsg = str(message.caption) + "\nFile contents: " + str(filetext).replace('\n\n','\n')
        response = get_response(message,usermsg)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    con = sqlite3.connect(os.path.join(script_dir,'chatter.db'))
    cursor = con.cursor()
    subscription = cursor.execute("select * from subscribers where user_id = ? and service = ?",(message.from_user.id,'AICHAT')).fetchall()
    if len(subscription)>0:
        try:
            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            filename = 'voice_' + str(message.from_user.id)
            with open(os.path.join(script_dir,'voices',filename + '.ogg'), 'wb') as new_file:
                new_file.write(downloaded_file)
            ogg_audio = AudioSegment.from_file(os.path.join(script_dir,'voices',filename + '.ogg'), format="ogg")
            ogg_audio.export(os.path.join(script_dir,'voices',filename + '.mp3'), format="mp3")
            transcript = AIClient.audio.transcriptions.create(model="whisper-1", file=open(os.path.join(script_dir,'voices',filename + '.mp3'),'rb'))
            response = get_response(message,transcript.text)
            bot.reply_to(message, response)
        except Exception as e:
            bot.reply_to(message, "Sorry, " + str(e))

@bot.message_handler()
def catch_all(message):
    if message.chat.type == 'private' or message.entities!=None:
        try:
            response = get_response(message,message.text)
            if "Response image:" in response:
                try:
                    resp_prompt = response.split("Response image:")
                    print("Got prompt:",resp_prompt[1])
                    response_image = AIClient.images.generate(
                        model="dall-e-3",
                        prompt=resp_prompt[1],
                        n=1,
                        size="1024x1024"
                    )
                    response = resp_prompt[0]
                    image_url = response_image.data[0].url
                    bot.send_photo(message.chat.id, image_url)
                except Exception as e:
                    bot.reply_to(message, "Sorry, " + str(e))
            if len(response):
                bot.reply_to(message, response)
        except Exception as e:
            bot.reply_to(message, "Sorry, " + str(e))
    else:
        pass

update_db()
bot.infinity_polling(timeout=150,long_polling_timeout=150)
