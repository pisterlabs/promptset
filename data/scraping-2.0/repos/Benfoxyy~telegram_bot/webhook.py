from flask import Flask, request
import jdatetime,datetime
import openai
import telepot
import urllib3
from mtranslate import translate
from random import randrange
import requests
from api_key import api
from TKN import t
import cost_information
from req_list import *

proxy_url = "http://proxy.server:3128"
telepot.api._pools = {
    'default': urllib3.ProxyManager(proxy_url=proxy_url, num_pools=3, maxsize=10, retries=False, timeout=30),
}
telepot.api._onetime_pool_spec = (urllib3.ProxyManager, dict(proxy_url=proxy_url, num_pools=1, maxsize=1, retries=False, timeout=30))

secret = "BOT"
bot = telepot.Bot(t)
bot.setWebhook("https://BinBin.pythonanywhere.com/{}".format(secret), max_connections=1)

app = Flask(__name__)

def weather(city,key,chat_id):
    try:
        URL="https://api.openweathermap.org/data/2.5/weather"
        PARAMS={'q':city,'appid':key}
        r=requests.get(url=URL,params=PARAMS)
        data=r.json()
        name=data['name']
        tm=data['main']['temp']
        hmt=data['main']['humidity']
        Condition=data['weather'][0]['description']
        country=data['sys']['country']
        min_temp=data['main']['temp_min']
        max_temp=data['main']['temp_max']
        wind_speed=data['wind']['speed']
        wind_deg=data['wind']['deg']
        sunrise=str(datetime.datetime.fromtimestamp(data['sys']['sunrise']+12700))
        sunset=str(datetime.datetime.fromtimestamp(data['sys']['sunset']+12650))
        ctime = str(datetime.datetime.fromtimestamp(data['dt']+12650))

        return(
            f'''
🌍Information :

Country : {country}
City name : {name}
Time : {ctime[11:16]}

⛅️Weather :

Condition : {Condition}
Temperature : {"%.2f" %(float(tm-272.15))}C
Minimum Temperature : {"%.2f" %(float(min_temp-272.15))}C
Maximum Temperature : {"%.2f" %(float(max_temp-272.15))}C
Humidity : {hmt}%

☀️Sunrise/set :

Sunrise : {sunrise[11:16]}
Sunset : {sunset[11:16]}

🌪Wind :

Wind speed : {wind_speed}km/h
Wind degree : {wind_deg}
            '''
        )
    except:
        bot.sendMessage(chat_id,'Something went wrong. Please try again later')

def ask(question):
    openai.api_key=(api)

    prompt=question

    respon = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        temperature=0.4,
        max_tokens=64
    )
    return (str(respon['choices'][0]['text']))

def choice(number):
    rn=randrange(1,(number+1))
    return rn

def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)

    if content_type == 'text':
        txt=((msg['text'])[10:])
        if msg['text']=='/start':
            bot.sendMessage(chat_id,'for seeing all commands , please use /help')
        
        elif msg['text']=='/price':
            try:
                s_price=list(map(cost_information.cost,(lis)))
                f_price=cost_information.design(lst=s_price)
                bot.sendMessage(chat_id,f_price)
            except:
                bot.sendMessage(chat_id,'Something went wrong. Please try again later')

        elif (msg['text'])[:10]=='/translate':
            if msg['text']=='/translate':
                bot.sendMessage(chat_id,'please write your text after /translate<here>')
            else:
                tr=translate(txt,'en')
                bot.sendMessage(chat_id,tr)

        elif msg['text']=='/chat_id':
            bot.sendMessage(chat_id,chat_id)
        elif (msg['text'])[:6]=='/guess':
            if msg['text']=='/guess':
                bot.sendMessage(chat_id,'please write your number after /guess<here>')
            else:
                chance=(msg['text'])[6:]
                bot.sendMessage(chat_id,choice(number=int(chance)))
        elif (msg['text'])[:4]=='/ask':
            if msg['text']=='/ask':
                bot.sendMessage(chat_id,'please write your text after /ask<here>')
            else:
                question=((msg['text'])[5:])
                bot.sendMessage(chat_id,ask(question=question))
        elif msg['text']=='/date':
            s=str(jdatetime.date.today()).replace('-','/')
            ad=str(datetime.date.today()).replace('-','/')
            bot.sendMessage(chat_id,f'☀️{s}☀️\n- - - - - - - - - - - - - - -\n🎄{ad}🎄')
        elif msg['text'][:8]=='/weather':
            if msg['text']=='/weather':
                bot.sendMessage(chat_id,'please write your city name after /weather<here>')
            else:
                try:
                    key='3f591382e9e21b0549d3fbdeeca1bf13'
                    city=msg['text'][9:]
                    bot.sendMessage(chat_id,weather(city=city,key=key,chat_id=chat_id))
                except:
                    bot.sendMessage(chat_id,'Something went wrong. Please try again later')
        elif msg['text']=='/help':
            bot.sendMessage(chat_id,
                            '''
commands list :

⚙️/star:
start Robobot

🔍/help:
see every command

🤖/ask <your question>:
Chat GPT

🈹/translate <your text>:
translate any language to English

🎲/guess <chance>:
I guess a number
between the number what you said

🗓/date:
Solar
A.D

⛅️/weather <city>
Current weather

🪙/price:
Gold,Coin,Currency

🆔/chat_id:
get your chat id
                            '''
                            )

@app.route('/{}'.format(secret), methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    if "message" in update:
        handle(update['message'])
    return "OK"