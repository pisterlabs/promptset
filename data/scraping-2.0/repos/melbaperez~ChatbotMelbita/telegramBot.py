from flask import Flask,request
import requests
import openai
import os
import datetime
import logging
from vectorDatabase import VectorDatabase
from utils import readYaml
from functionsCompletions import *

key  =  os.environ["TELEGRAM_CHATBOT_API_KEY"]
openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"] 
openai.api_key = os.environ["OPENAI_API_KEY"]

app = Flask(__name__)
milvusVectorDatabase = VectorDatabase()

CATEGORY_COINDE = "COINDE"
CATEGORY_ELIMINAR = "ELIMINAR_CHAT"
NOT_INFO_FLAG = 0
INFO_FLAG = 1
MAX_TIME_DELAY = 10200


def sendMessage(chatId, answer):
    ''' Sends a message via the Telegram API. '''
    url = f"https://api.telegram.org/bot{key}/sendMessage"

    payload = {
            "text": answer,
            "chat_id": chatId
            }
    response = requests.get(url,params=payload)
    if not response.ok:
        logging.warning(f"TELEGRAM: Fallo en el envío de mensaje - {response.status_code}")


def setMessageTypingChatbot(chatId):
    ''' Simulates the typing action of a chatbot in Telegram. '''
    url = f"https://api.telegram.org/bot{key}/sendChatAction"

    payload = {
            "action": "typing",
            "chat_id": chatId
            }
    response = requests.post(url, json=payload)
    if not response.ok:
        logging.warning(f"TELEGRAM: Fallo en la acción escribiendo del chatbot - {response.status_code}")


def sendMessageCallbackQuery(callbackQuery, messageButton):
    ''' Sends a response to a callback button in Telegram. '''
    url = f"https://api.telegram.org/bot{key}/answerCallbackQuery"  
    payload = {
            "callback_query_id": callbackQuery["id"],
            "text": messageButton
            }
    
    response = requests.post(url, json=payload)
    if not response.ok:
        logging.warning(f"TELEGRAM: Fallo al enviar la respuesta de los botones - {response.status_code}")


def sendMessageWithButtons(chatId):
    ''' Sends a message with buttons via the Telegram API. '''
    url = f"https://api.telegram.org/bot{key}/sendMessage"  
    
    buttons = [
        [{"text": "SÍ", "callback_data": "yes"}],
        [{"text": "NO", "callback_data": "no"}]
    ]
    
    payload = {
        "chat_id": chatId,
        "text": "Creo interpretar que lo que solicitas es que elimine todo el contenido del chat e información almacenada. Si esto es así, ya no recordaré nada sobre ti. ¿Deseas que elimine toda tu información y conversaciones anteriores?",
        "reply_markup": {"inline_keyboard": buttons}
    }
    
    response = requests.post(url, json=payload)
    if response.ok:
        logging.info("ELIMINAR: Se envía al usuario el mensaje de si desea eliminar la conversación")
    else:
        logging.warning(f"TELEGRAM: Error al enviar el mensaje con botones - {response.status_code}")


def handleButtonResponse(callbackQuery, chatId):
    ''' Handles the response of the buttons. '''
    if callbackQuery['data'] == "yes":
        milvusVectorDatabase.deleteUserMessages(chatId)
        messageButton = "Conversación eliminada correctamente"
        messageToSend = "Sus datos han sido eliminada de nuestra base de datos. ¿En qué más puedo ayudarte?"
        logging.info("ELIMINAR: se elimina la conversación de {chatId}")
    else:
        messageButton = "Conversación no eliminada"
        messageToSend = "Aseguramos su privacidad y en cualquier momento que desee eliminar sus datos de la base de datos tan solo debe indicarlo. ¿En qué más puedo ayudarte?"
    
    sendMessageCallbackQuery(callbackQuery, messageButton)
    sendMessage(chatId, messageToSend)


def generateReponse(chatId, userInput, date):
    ''' Generates a response from the user's input. '''

    categorySentence = getCategorySentence(userInput)
        
    if categorySentence == CATEGORY_ELIMINAR:
        sendMessageWithButtons(chatId)
    else:
        if categorySentence == CATEGORY_COINDE:
            milvusVectorDatabase.insertData(f"Usuario: {userInput}", INFO_FLAG, date, chatId)
        else:
            milvusVectorDatabase.insertData(f"Usuario: {userInput}", NOT_INFO_FLAG, date, chatId)

        listDataLastMessages = milvusVectorDatabase.getLastMessages(chatId)
        memorySentences = milvusVectorDatabase.searchInformation(userInput, chatId, listDataLastMessages[1])
        responseChatbot = completion(listDataLastMessages[0], memorySentences)
        
        logging.info(f"\n {memorySentences}")
        logging.info(f"RESPUESTA: {responseChatbot}")

        milvusVectorDatabase.insertData(f"IA: {responseChatbot}", NOT_INFO_FLAG, date+1, chatId) 
        sendMessage(chatId, responseChatbot)


@app.route("/",methods=["POST","GET"])
def index():
    ''' Processes Telegram requests and performs actions such as sending messages and generating replies. '''
    if(request.method == "POST"):
        resp = request.get_json()

        if "message" in resp: 
            if "text" in resp["message"]:
                msgUser = resp["message"]["text"]
                senderName = resp["message"]["from"]["first_name"]
                chatId = resp["message"]["chat"]["id"]
                date = resp["message"]["date"]

                messageTime = datetime.datetime.utcfromtimestamp(date).strftime('%Y%m%d%H%M%S')
                nowTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                logging.info(f"-------------------------------- ACCESO USUARIO: {chatId} {senderName} --------------------------------")
                logging.info(f"Tiempo entre recibo y envío de mensajes: {int(nowTime) - int(messageTime)} segundos")
                
                if int(nowTime) - int(messageTime) < MAX_TIME_DELAY:
                    
                    if milvusVectorDatabase.isNewUser(chatId):
                        logging.info(".....NUEVO USUARIO.....")
                        milvusVectorDatabase.insertData(f"Usuario: Mi nombre es {senderName}", INFO_FLAG, date-1, chatId)
                    
                    setMessageTypingChatbot(chatId)

                    if msgUser == "/start":
                        sendMessage(chatId, f"Bienvenid@ {senderName}, soy Melbita tu asistente personal, ¿en qué puedo ayudarte?")
                    else:
                        generateReponse(chatId, msgUser, date)

        elif "callback_query" in resp:
            callbackQuery = resp["callback_query"]
            handleButtonResponse(callbackQuery, resp["callback_query"]["from"]["id"])

    return "Done"
    

@app.route("/setwebhook/")
def setwebhook():
    ''' Set a webhook to receive Telegram updates in a web application. '''
    url = readYaml("data/configuration.yaml")['APP']['URL_NGROK']
    setWebhookResponse = requests.get(f"https://api.telegram.org/bot{key}/setWebhook?url={url}")
    if setWebhookResponse:
        logging.info("Webhook configurado con éxito")
        return "yes"
    else:
        logging.warning(f"Webhook no configurado con éxito: {setWebhookResponse.text}")
        return "fail"


logging.basicConfig(filename='log/app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', encoding="utf-8")
#setwebhook()    
