import requests
import sett
import json
import time
import openai
import csv
import os
from datetime import datetime

# chatgpt
openai.api_key = os.getenv("OPENAI_API_KEY")


def obtener_Mensaje_whatsapp(message):
    if 'type' not in message:
        text = 'mensaje no reconocido'
        return text

    typeMessage = message['type']
    if typeMessage == 'text':
        text = message['text']['body']
    elif typeMessage == 'button':
        text = message['button']['text']
    elif typeMessage == 'interactive' and message['interactive']['type'] == 'list_reply':
        text = message['interactive']['list_reply']['title']
    elif typeMessage == 'interactive' and message['interactive']['type'] == 'button_reply':
        text = message['interactive']['button_reply']['title']
    else:
        text = 'mensaje no procesado'

    return text


def enviar_Mensaje_whatsapp(data):
    try:
        whatsapp_token = os.getenv("WHATSAPP_TOKEN")
        whatsapp_url = os.getenv("WHATSAPP_URL")
        headers = {'Content-Type': 'application/json',
                   'Authorization': 'Bearer ' + whatsapp_token}
        response = requests.post(whatsapp_url,
                                 headers=headers,
                                 data=data)

        print("response ->", response)
        if response.status_code == 200:
            return 'mensaje enviado', 200
        else:
            return 'error al enviar mensaje', response.status_code
    except Exception as e:
        return e, 403


def text_Message(number, text):
    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "text",
            "text": {
                    "body": text
            }
        }
    )
    return data


def buttonReply_Message(number, options, body, footer, sedd, messageId):
    buttons = []
    for i, option in enumerate(options):
        buttons.append(
            {
                "type": "reply",
                "reply": {
                    "id": sedd + "_btn_" + str(i+1),
                    "title": option
                }
            }
        )

    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {
                    "text": body
                },
                "footer": {
                    "text": footer
                },
                "action": {
                    "buttons": buttons
                }
            }
        }
    )
    return data


def listReply_Message(number, options, body, footer, sedd, messageId):
    rows = []
    for i, option in enumerate(options):
        rows.append(
            {
                "id": sedd + "_row_" + str(i+1),
                "title": option,
                "description": ""
            }
        )

    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {
                    "text": body
                },
                "footer": {
                    "text": footer
                },
                "action": {
                    "button": "Ver Opciones",
                    "sections": [
                        {
                            "title": "Secciones",
                            "rows": rows
                        }
                    ]
                }
            }
        }
    )
    return data


def document_Message(number, url, caption, filename):
    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "document",
            "document": {
                "link": url,
                "caption": caption,
                "filename": filename
            }
        }
    )
    return data


def sticker_Message(number, sticker_id):
    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "sticker",
            "sticker": {
                "id": sticker_id
            }
        }
    )
    return data


def get_media_id(media_name, media_type):
    media_id = ""
    if media_type == "sticker":
        media_id = sett.stickers.get(media_name, None)
    elif media_type == "image":
        media_id = sett.images.get(media_name, None)
    elif media_type == "video":
        media_id = sett.videos.get(media_name, None)
    elif media_type == "audio":
        media_id = sett.audio.get(media_name, None)
    return media_id


def replyReaction_Message(number, messageId, emoji):
    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "type": "reaction",
            "reaction": {
                "message_id": messageId,
                "emoji": emoji
            }
        }
    )
    return data


def replyText_Message(number, messageId, text):
    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": number,
            "context": {"message_id": messageId},
            "type": "text",
            "text": {
                "body": text
            }
        }
    )
    return data


def markRead_Message(messageId):
    data = json.dumps(
        {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id":  messageId
        }
    )
    return data


def administrar_chatbot(text, number, messageId, name):
    text = text.lower()  # mensaje que envio el usuario
    list = []

    markRead = markRead_Message(messageId)
    list.append(markRead)
    time.sleep(2)

    while text != "es todo":
        if "hola" in text:
            body = "¡Hola! 👋 Bienvenido a Bigdateros. ¿Cómo podemos ayudarte hoy?"
            footer = "Equipo Bigdateros"
            options = ["✅ servicios", "📅 agendar cita"]

            replyButtonData = buttonReply_Message(
                number, options, body, footer, "sed1", messageId)
            replyReaction = replyReaction_Message(number, messageId, "🫡")
            list.append(replyReaction)
            list.append(replyButtonData)
        elif "servicios" in text:
            body = "Tenemos varias áreas de consulta para elegir. ¿Cuál de estos servicios te gustaría explorar?"
            footer = "Equipo Bigdateros"
            options = ["Analítica Avanzada",
                       "Migración Cloud", "Inteligencia de Negocio"]

            listReplyData = listReply_Message(
                number, options, body, footer, "sed2", messageId)
            sticker = sticker_Message(
                number, get_media_id("perro_traje", "sticker"))

            list.append(listReplyData)
            list.append(sticker)
        elif "inteligencia de negocio" in text:
            body = "Buenísima elección. ¿Te gustaría que te enviara un documento PDF con una introducción a nuestros métodos de Inteligencia de Negocio?"
            footer = "Equipo Bigdateros"
            options = ["✅ Sí, envía el PDF.", "⛔ No, gracias"]

            replyButtonData = buttonReply_Message(
                number, options, body, footer, "sed3", messageId)
            list.append(replyButtonData)
        elif "sí, envía el pdf" in text:
            sticker = sticker_Message(
                number, get_media_id("pelfet", "sticker"))
            textMessage = text_Message(
                number, "Genial, por favor espera un momento.")

            enviar_Mensaje_whatsapp(sticker)
            enviar_Mensaje_whatsapp(textMessage)
            time.sleep(3)

            document = document_Message(
                number, sett.document_url, "Listo 👍🏻", "Inteligencia de Negocio.pdf")
            enviar_Mensaje_whatsapp(document)
            time.sleep(3)

            body = "¿Te gustaría programar una reunión con uno de nuestros especialistas para discutir estos servicios más a fondo?"
            footer = "Equipo Bigdateros"
            options = ["✅ Sí, agenda reunión", "No, gracias."]

            replyButtonData = buttonReply_Message(
                number, options, body, footer, "sed4", messageId)
            list.append(replyButtonData)
        elif "sí, agenda reunión" in text:
            body = "Estupendo. Por favor, selecciona una fecha y hora para la reunión:"
            footer = "Equipo Bigdateros"
            options = ["📅 10: mañana 10:00 AM",
                       "📅 7 de junio, 2:00 PM", "📅 8 de junio, 4:00 PM"]

            listReply = listReply_Message(
                number, options, body, footer, "sed5", messageId)
            list.append(listReply)
        elif "no, gracias." in text:
            textMessage = text_Message(
                number, "Perfecto! No dudes en contactarnos si tienes más preguntas. Recuerda que también ofrecemos material gratuito para la comunidad. ¡Hasta luego! 😊")
            list.append(textMessage)
            break
        elif "conectar con personal" in text:
            buttonData = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": number,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {
                        "text": "Entendido. Haz clic en el botón a continuación para conectarte con nuestro personal real."
                    },
                    "action": {
                        "buttons": [
                            {
                                "type": "url",
                                "url_button": {
                                    "title": "Conectar con Personal Real",
                                    # reemplaza esto con el enlace a la página de conexión con personal real
                                    "url": "https://link-to-connection-page.com"
                                }
                            }
                        ]
                    }
                }
            }
            data = json.dumps(buttonData)
            enviar_Mensaje_whatsapp(data)
            break
        else:
            data = text_Message(
                number, "Lo siento, no entendí lo que dijiste. ¿Quieres que te ayude con alguna de estas opciones?")
            list.append(data)

    for item in list:
        enviar_Mensaje_whatsapp(item)


def generar_respuesta_chatgpt(user_message, number, espedido=False):
    messages = [{'role': 'system', 'content': """
                Soy Sari, tu asistente virtual de Joyas Boulevard en Argentina. Estoy aquí para ayudarte a explorar nuestras colecciones de joyas y responder a todas tus preguntas. Si en algún momento prefieres hablar con uno de nuestros expertos en joyería, solo tienes que decírmelo. La página de la E-commerce es joyasboulevard.com y el Instagram es @joyeriaboluevard. Soy muy amable.
                """}]

    historial = get_chat_from_csv(number)
    messages.extend(historial)

    messages.append({'role': 'user', 'content': user_message})

    if espedido:
        messages.append(
            {'role': 'system', 'content': 'Crea un resumen del pedido anterior en formato JSON. \
            Analiza la lista de productos de la joyería ingresada al inicio y compara con el pedido del usuario. \
            Solo cuando hayas analizado el pedido completo del usuario, categorízalo en lista de anillos, lista de pulseras, lista de pendientes, etc. \
            Los campos del json deben ser 1) lista de anillos con atributos de nombre, tamaño, cantidad, 2) lista de pulseras con atributos de nombre, tamaño, cantidad, \
            3) lista de pendientes con atributos de nombre, tamaño, cantidad, etc. \
            Luego, actualiza el precio total del pedido una vez que hayas listado cada ítem.'},
        )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8
    )

    print("response", response.choices[0].message["content"])
    return response.choices[0].message["content"]


def guardar_conversacion(conversation_id, number, name, user_msg, timestamp, bot_msg=''):
    try:
        conversations = []
        conversation = [conversation_id, number, name,
                        user_msg, bot_msg, datetime.fromtimestamp(timestamp)]
        # Guardar las conversaciones en el archivo CSV
        with open('conversaciones.csv', 'a', newline='') as csv_file:
            data = csv.writer(csv_file, delimiter=',')
            data.writerow(conversation)

        messages = get_chat_from_csv(number)
    except Exception as e:
        return e, 403


def get_chat_from_csv(number):
    messages = []
    with open('conversaciones.csv') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['number'] == number:
                print('number')
                user_msg = {'role': 'user', 'content': row['user_msg']}
                bot_msg = {'role': 'assistant', 'content': row['bot_msg']}
                messages.append(user_msg)
                messages.append(bot_msg)
    return messages


def guardar_pedido(jsonPedido, number):
    # Eliminar el texto que sigue al JSON
    start_index = jsonPedido.find("{")
    end_index = jsonPedido.rfind("}")

    # Extrae la cadena JSON de la respuesta
    json_str = jsonPedido[start_index:end_index+1]

    # Convierte la cadena JSON en un objeto de Python
    pedido = json.loads(json_str)

    # Ahora puedes usar 'pedido' como un objeto de Python
    with open('pedidos.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        anillos = [
            f"{anillo['cantidad']} {anillo['nombre']} - {anillo['precio']} pesos" for anillo in pedido['anillos']]
        pulseras = [
            f"{pulsera['cantidad']} {pulsera['nombre']} - {pulsera['precio']} pesos" for pulsera in pedido['pulseras']]
        pendientes = [
            f"{pendiente['cantidad']} {pendiente['nombre']} - {pendiente['precio']} pesos" for pendiente in pedido['pendientes']]

        writer.writerow([number,
                         ', '.join(anillos),
                         ', '.join(pulseras),
                         ', '.join(pendientes),
                         pedido['precio_total'],
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
