import telebot
import json
import openai
import os
import asyncio

telegram_token = os.getenv('TELEGRAMTOKEN')
openai_token = os.getenv( 'OPENAITOKEN' )

bot = telebot.TeleBot( telegram_token )
#openai.api_key = openai_token

portais = {
        '0': ['Globo/G1', 'https://g1.globo.com' ],
        '1': ['BBC Brazil','https://www.bbc.com/portuguese' ],
        '2': [ 'CNN Brazil', 'https://www.cnnbrasil.com.br/']
        }

cmds = {
        'start':'',
        'news': '',
        'info': '',
        'search': '',
        'help': ''
        }
# Recurso em implementação
# Objetivo criar um uma tarefa agendada para execução em um horario especifico, diariamente.
async def a_chsendnews(delay):
	await await asyncio.sleep( delay )


async def agtask_sendnews():
	await a_achsendnews( t )
# rodar o bot.polling e agtask_sendnews dentro de um asyncio (vide asyncio.gather functio)
#-------------------------------------Agendamento -----------------------

def process(text):
    response = openai.Completion.create(
            model='text-davinci-003',
            prompt=text,
            temperature=0.5,
            max_tokens=600,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=['you:']
            )
    return response['choices'][0]['text']


@bot.channel_post_handler(commands=['news'])
def ch_send_news( message ):
    import datetime
    date = datetime.date.today()
    date = str( date.day ) + '-' + str(date.month) + '-' + str(date.year)
    datares = None
    with open(f'/home/opc/news-crawler/datanews/Noticias-{date}.json', 'r') as fl:
        datares = fl.read()
        fl.close()
    out = json.loads(datares)
    #print(out[0])
    import random
    out = random.sample(out, k=3)
    for i in out:
        bot.reply_to(message, i['title']+'\n'+i['href'])

@bot.channel_post_handler(commands=['info'])
def ch_send_info( message ):
    text = """
        Chat bot e visualizador de noticias.
        Desenvolvido por wsricardo.
        Site: www.github.com/wsricardo
        Blog: https://wsricardo.blogspot.com
    """
    bot.reply_to( message, text )


@bot.channel_post_handler(commands=['portais'])
def ch_send_portais( message ):
    text = '\n'.join([ ' '.join( portais[i] ) for i in portais.keys() ] )
    bot.reply_to( message, text )

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to( message, 'Bem vindo ao chatbot de noticias Sofia. Acompanhe as principais noticias em portais como CNN, BBC, G1/Globo.')

@bot.message_handler( commands=['news', 'help'] )
def send_news(message):
    import datetime
    date = datetime.date.today()
    date = str(date.day) + '-' + str(date.month) + '-' + str(date.year)
    datares = None
    with open(f'/home/opc/news-crawler/datanews/Noticias-{date}.json', 'r') as fl:
        datares = fl.read()
        fl.close()
    out = json.loads(datares)
    #print(out[0])
    import random
    out = random.sample(out, k=3)
    for i in out:
        bot.reply_to(message, i['title']+'\n'+i['href'])

@bot.message_handler( commands=['portais', 'help'] )
def send_portais( message ):
    text = '\n'.join([ ' '.join( portais[i] ) for i in portais.keys() ] )
    bot.reply_to( message, text )

@bot.message_handler( commands=['info', 'help' ] )
def send_info( message ):
    text = """
        Chat bot e visualizador de noticias.
        Desenvolvido por wsricardo.
        Site: www.github.com/wsricardo
        Blog: https://wsricardo.blogspot.com
    """
    bot.reply_to( message, text )

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    text = message.text
    test = lambda text: 'Noticia' in text or 'noticia' in text or 'Notícia' in text or 'Noticias' in text or 'noticias' in text or 'Notícias' in text
    
    if test(text):
        datares = None
        with open('noticias.json', 'r') as fl:
            datares = fl.read()
            fl.close()
        out = json.loads(datares)
        #print(out[0])
        import random
        out = random.sample(out, k=3)
        for i in out:
            bot.reply_to(message, i['title']+'\n'+i['href'])
    else:
        bot.reply_to(message, process(text))


bot.infinity_polling()
