import configparser
import os
import time
import random
import sqlite3
import requests
import sys
from typing import Tuple
# opcional
#sys.path.append('/home/morfetico/.local/lib/python3.11/site-packages/')
#
import telebot
import shutil
import openai
import logging
import urllib.parse

start_time = time.time()
request_count = 0


# Configuração do log
# Configuração do log
logging.basicConfig(filename='bot-telegram.log', level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info("Bot iniciado.")  # Ao iniciar o bot


# rate limit
REQUEST_LIMIT = 20  # número máximo de pedidos em um intervalo de tempo
TIME_WINDOW = 60  # intervalo de tempo em segundos
request_count = 0  # contador para pedidos feitos
start_time = time.time()  # tempo em que começamos a contar os pedidos


#shutil.move(src_file, dst_file)

def create_table():
    # Conectando ao banco de dados
    with sqlite3.connect('frases.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS frases (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            frase TEXT NOT NULL
        );
        """)

def insert_frase(frase:str):
    with sqlite3.connect('frases.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO frases (frase) VALUES (?)", (frase,))

def get_random_frase()->str:
    with sqlite3.connect('frases.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT frase FROM frases ORDER BY RANDOM() LIMIT 1")
        return cursor.fetchone()[0]

create_table()

config1 = configparser.ConfigParser()
config1.read('token-telegram.cfg')
TOKEN = config1['DEFAULT']['TOKEN']
bot = telebot.TeleBot(TOKEN)

config2 = configparser.ConfigParser()
config2.read('token-openai.cfg')
openai_key = config2['DEFAULT']['API_KEY']
openai.api_key = openai_key

# xingamentos
@bot.message_handler(commands=['xinga'])
def random_message(message):
    logging.info("Comando /xinga chamado.")
    conn = sqlite3.connect('frases.db')
    c = conn.cursor()
    c.execute("SELECT frase FROM frases")
    frases = c.fetchall()
    conn.close()
    if not frases:
        bot.send_message(message.chat.id, 'Não há frases cadastradas.')
    else:
        frase_escolhida = random.choice(frases)[0]
        if message.reply_to_message:
            bot.reply_to(message.reply_to_message, "@{} {}".format(message.reply_to_message.from_user.username, frase_escolhida))
        elif len(message.text.split()) > 1 and message.text.split()[1].startswith('@'):
            username = message.text.split()[1][1:]
            bot.send_message(message.chat.id, "@{} {}".format(username, frase_escolhida))
        else:
            bot.send_message(message.chat.id, frase_escolhida)
#
# Chat GPT
# Chat GPT

# Importa configuração do prompt
def get_prompt(text_message):
    with open('prompt.cfg', 'r') as arquivo:
        base_prompt = arquivo.read().strip()
    return base_prompt + " " + text_message.replace("@" + bot.get_me().username, "").strip() + "."

@bot.message_handler(func=lambda message: message.text is not None and (bot.get_me().username in message.text or message.reply_to_message is not None))
def responder(message):
    chat_type = message.chat.type
    chat_title = message.chat.title if chat_type != 'private' else None
    username = message.from_user.username if message.from_user.username else message.from_user.first_name
    
    logging.info("----------")  # Separate the logs
    if chat_title:
        logging.info(f"Mensagem recebida de @{username} no grupo '{chat_title}': {message.text}")
    else:
        logging.info(f"Mensagem recebida em pvt de @{username}: {message.text}")

    # Proíbe mensagens enviadas diretamente ao bot
    if chat_type == 'private':
        return

    text_prompt = get_prompt(message.text)  # passando message.text como argumento

    # Verifica se a mensagem é uma resposta a outra mensagem
    if message.reply_to_message:
        # Mensagem original (a que foi respondida)
        original_msg_text = message.reply_to_message.text

        # Mensagem de resposta (a atual)
        reply_msg_text = message.text

        # Combina as duas mensagens para análise
        combined_text = f"Mensagem Original: {original_msg_text}\nResposta: {reply_msg_text}"

        # Atualiza o prompt para usar o texto combinado
        text_prompt = get_prompt(combined_text)
    else:
        # Se não for uma resposta, usa o texto da mensagem como está
        text_prompt = get_prompt(message.text)

    # limitação de taxa
    global contador_requisicoes, tempo_inicial

    # Se a message for de um bot, não processamos
    if message.from_user.is_bot:
        return

    # Verificando o limite de requisições
    global start_time, request_count
    # Verificando o limite de requisições
    current_time = time.time()
    if current_time - start_time > TIME_WINDOW:
        start_time = current_time
        request_count = 0
    if request_count > REQUEST_LIMIT:
        bot.send_message(chat_id=message.chat.id, text="Estou recebendo muitos pedidos. Por favor, tente novamente mais tarde.")
        return
    request_count += 1

    # Se a message é uma resposta para o nosso bot
    if (message.reply_to_message and message.reply_to_message.from_user.username == bot.get_me().username) or (bot.get_me().username in message.text):
        text_prompt = get_prompt(message.text) 
    else:
        return

    try:
        start_time = time.time()
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=text_prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.9,
        ).choices[0].text
        response_time = time.time() - start_time

        # Se a resposta estiver vazia, tenta novamente
        while not response:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=text_prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.9,
            ).choices[0].text
    except (openai.OpenAIError, requests.exceptions.RequestException) as e:
        response = "Desculpe, ocorreu um erro ao me conectar à API do OpenAI. Por favor, tente novamente mais tarde."

    # Envio da resposta
    if message.reply_to_message:
        bot.reply_to(message.reply_to_message, response)
    else:
        bot.send_message(chat_id=message.chat.id, text=response)



# Adiciona o comando de busca no YouTube
@bot.message_handler(commands=['youtube'])
def youtube_search_command(message):
    query = message.text.replace("/youtube", "").strip()

    if not query:
        bot.send_message(chat_id=message.chat.id, text="Por favor, execute o /youtube com algum termo de busca")
        return

    API_KEY = open("token-google.cfg").read().strip() # API Key do Google
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&key={API_KEY}"

    try:
        results = requests.get(search_url).json()
        if results.get("pageInfo").get("totalResults") == 0:
            bot.send_message(chat_id=message.chat.id, text="Não foram encontrados resultados para a sua pesquisa.")
            return
        results = results["items"]
        response = "Resultados da pesquisa no YouTube para '" + query + "': \n"
        for result in results[:5]:
            response += result["snippet"]["title"] + " - https://www.youtube.com/watch?v=" + result["id"]["videoId"] + "\n"
    except (requests.exceptions.RequestException, KeyError) as e:
        response = "Desculpe, ocorreu um erro ao acessar a API do YouTube. Por favor, tente novamente mais tarde."

    bot.send_message(chat_id=message.chat.id, text=response)

# Adiciona o comando de busca usando o Google
@bot.message_handler(commands=['search'])
def search_command(message):
    query = message.text.replace("/search", "").strip()

    if not query:
        bot.send_message(chat_id=message.chat.id, text="Por favor, execute o /search com algum termo de busca")
        return

    API_KEY = open("token-google.cfg").read().strip() # API Key do Google
    SEARCH_ENGINE_ID = open("token-google-engine.cfg").read().strip() # API Key do Google
    search_url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"

    try:
        results = requests.get(search_url).json()
        if results.get("searchInformation").get("totalResults") == "0":
            bot.send_message(chat_id=message.chat.id, text="Não foram encontrados resultados para a sua pesquisa.")
            return
        results = results["items"]
        response = "Resultados da pesquisa para '" + query + "': \n"
        for result in results[:5]:
            response += result["title"] + " - " + result["link"] + "\n"
    except (requests.exceptions.RequestException, KeyError) as e:
        response = "Desculpe, ocorreu um erro ao acessar a API do Google. Por favor, tente novamente mais tarde."

    bot.send_message(chat_id=message.chat.id, text=response)



@bot.message_handler(commands=['real'])
def reais_message(message):
    bot.send_message(message.chat.id, 'O real não vale nada, é uma bosta essa moeda estatal de merda!')

@bot.message_handler(commands=['euro'])
def euro_message(message):
    url = 'https://economia.awesomeapi.com.br/all/EUR-BRL'
    r = requests.get(url)
    euro_data = r.json()
    valor_euro = euro_data['EUR']['bid']
    bot.send_message(message.chat.id, 'O valor atual do euro em reais é R$ ' + valor_euro)

@bot.message_handler(commands=['dolar'])
def dolar_message(message):
    url = 'https://economia.awesomeapi.com.br/all/USD-BRL'
    r = requests.get(url)
    dolar_data = r.json()
    valor_dolar = dolar_data['USD']['bid']
    bot.send_message(message.chat.id, 'O valor atual do dólar em reais é R$ ' + valor_dolar)

@bot.message_handler(commands=['btc'])
def bitcoin_price(message):
    url = "https://api.coincap.io/v2/assets/bitcoin"
    response = requests.get(url)
    data = response.json()
    price = round(float(data['data']['priceUsd']), 2)
    bot.send_message(message.chat.id, f'Cotação atual do Bitcoin em dolar: ${price}')

@bot.message_handler(commands=['xmr'])
def handle_btc(message):
    url = "https://api.coincap.io/v2/assets/monero"
    response = requests.get(url)
    data = response.json()
    price = round(float(data['data']['priceUsd']), 2)
    bot.send_message(message.chat.id, f'Cotação atual do Monero em dolar: ${price}')

@bot.message_handler(commands=['ajuda'])
def help_message(message):
    help_text = 'Comandos disponíveis:\n'
    help_text += '/add - Adiciona um xingamento, mas seja insolente por favor\n'
    help_text += '/list - Lista os xingamentos cadastrados\n'
    help_text += '/remover - Remove um xingamento\n'
    help_text += '/xinga - Envia um xingamento aleatório\n'
    help_text += '/dolar - Exibe a cotação do dolar em reais\n'
    help_text += '/euro - Exibe a cotação do euro em reais\n'
    help_text += '/btc - Exibe a cotação do Bitcoin em dolares\n'
    help_text += '/xmr - Exibe a cotação do Monero em dolares\n'
    help_text += '/real - Comando desnecessário pelo óbvio, mas tente executar pra ver...\n'
    help_text += '/youtube - Exibe resultados de busca de vídeos no Youtube\n'
    help_text += '/search - Exibe resultados de busca no Google'
    bot.send_message(message.chat.id, help_text)

@bot.message_handler(commands=['add'])
def add_message(message):
    text = message.text.split()
    if len(text) < 2:
        bot.send_message(message.chat.id, 'Comando inválido. Use /add e insira o xingamento')
        return
    frase = text[1]
    if len(message.text.split(' ', 1)[1]) > 150:
        bot.send_message(message.chat.id, 'Xingamento muito longo, por favor use até 150 caracteres')
        return
    conn = sqlite3.connect('frases.db')
    c = conn.cursor()
    frase = message.text.split(' ', 1)[1]
    c.execute("INSERT INTO frases (frase) VALUES (?)", (frase,))
    conn.commit()
    conn.close()
    bot.send_message(message.chat.id, 'Xingamento adicionado com sucesso! Seu zuero!')


@bot.message_handler(commands=['list'])
def list_message(message):
    chat_id = message.chat.id
    if message.chat.type != 'private':
        admin_ids = [admin.user.id for admin in bot.get_chat_administrators(chat_id) if admin.status != 'creator']
        owner_id = [admin for admin in bot.get_chat_administrators(chat_id) if admin.status == 'creator'][0].user.id
        if message.from_user.id == owner_id or message.from_user.id in admin_ids:
            conn = sqlite3.connect('frases.db')
            c = conn.cursor()
            c.execute("SELECT id, frase FROM frases")
            frases = c.fetchall()
            conn.close()
            if not frases:
                bot.send_message(message.chat.id, 'Não há frases cadastradas.')
            else:
                for frase in frases:
                    message_enviada = False
                    bot.send_message(message.chat.id, 'Xingamentos cadastrados:')
                    chunk_size = 20  # numero de frases por message
                    if not message_enviada:
                        for i in range(0, len(frases), chunk_size):
                            message = '\n'.join([f'{frase[0]}: {frase[1]}' for frase in frases[i:i+chunk_size]])
                            bot.send_message(message.chat.id, message)
                            time.sleep(5)
                            message_enviada = True
                    else:
                        message_enviada = True
                        break

                    message_enviada = True
                    break

        else:
            bot.send_message(message.chat.id, 'Apenas o administrador e o dono do grupo podem executar este comando')
    else:
        bot.send_message(message.chat.id, 'Este comando não pode ser usado em chats privados')


@bot.message_handler(commands=['remover'])
def remover_message(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    if message.chat.type != 'private':
        admin_ids = [admin.user.id for admin in bot.get_chat_administrators(chat_id) if admin.status != 'creator']
        owner_id = [admin for admin in bot.get_chat_administrators(chat_id) if admin.status == 'creator'][0].user.id
        if user_id == owner_id or user_id in admin_ids:
            frase_list = message.text.split()
            if len(frase_list) < 2:
                bot.send_message(message.chat.id, 'Insira um ID válido para remover')
                return
            frase_id = frase_list[1]
            if not frase_id.isdigit():
                bot.send_message(message.chat.id, 'Insira um ID válido para remover, ID é um número, seu MACACO!')
                return
            frase = frase_list[1]
            conn = sqlite3.connect('frases.db')
            c = conn.cursor()
            c.execute("DELETE FROM frases WHERE ID = ?", (frase,))
            conn.commit()
            conn.close()
            bot.send_message(message.chat.id, 'Xingamento removido com sucesso!')
        else:
            bot.send_message(message.chat.id, 'Somente o dono do grupo e administradores podem executar este comando.')
    else:
        bot.send_message(message.chat.id, 'Este comando não pode ser executado em conversas privadas.') 

bot.polling()
