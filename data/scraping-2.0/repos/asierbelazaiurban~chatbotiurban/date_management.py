import openai
import os
import requests
import json
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from datetime import datetime
import re

app = Flask(__name__)

# Crear el directorio de logs si no existe
if not os.path.exists('logs'):
    os.mkdir('logs')

# Configurar el manejador de logs para escribir en un archivo
log_file_path = 'logs/chatbotiurban.log'
file_handler = RotatingFileHandler(log_file_path, maxBytes=10240000, backupCount=1)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Usa DEBUG o INFO según necesites

# Añadir el manejador de archivos al logger de la aplicación
app.logger.addHandler(file_handler)

# También añadir un manejador de consola para la salida estándar
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
console_handler.setLevel(logging.DEBUG)  # Asegúrate de que este nivel sea consistente con file_handler.setLevel
app.logger.addHandler(console_handler)

# Establecer el nivel del logger de la aplicación
app.logger.setLevel(logging.DEBUG)

app.logger.info('Inicio de la aplicación ChatbotIUrban')

def get_openai_response(texto, fecha_actual):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables")

    instruccion_gpt4 = ("Tu tarea es identificar las referencias temporales en la pregunta del usuario y convertirlas a fechas específicas en formato MySQL, utilizando la fecha actual como referencia. Ejemplos de referencias temporales incluyen 'dentro de dos días', 'el finde', 'la semana que viene', 'mañana', 'en 2024', etc. La fecha actual es: " + fecha_actual.strftime("%Y-%m-%d"))

    respuesta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": instruccion_gpt4},
            {"role": "user", "content": texto},
        ]
    )

    respuesta_texto = respuesta.choices[0].message['content']
    app.logger.info("Respuesta de OpenAI: %s", respuesta_texto)

    # Extraer fechas del texto
    fechas_encontradas = re.findall(r'\d{4}-\d{2}-\d{2}', respuesta_texto)
    app.logger.info("Fechas encontradas en el texto: %s", fechas_encontradas)

    return fechas_encontradas

def asegurarse_el_formato_mysql(texto, fecha_actual):
    fechas_encontradas = get_openai_response(texto, fecha_actual)
    app.logger.info("Fechas encontradas: %s", fechas_encontradas)

    fecha_inicial = fecha_final = None
    if fechas_encontradas:
        fecha_inicial = fechas_encontradas[0]
        fecha_final = fechas_encontradas[1] if len(fechas_encontradas) > 1 else fecha_inicial

    app.logger.info("Fecha inicial (MySQL): %s", fecha_inicial)
    app.logger.info("Fecha final (MySQL): %s", fecha_final)

    return fecha_inicial, fecha_final

def obtener_eventos(pregunta, chatbot_id):
    fecha_actual = datetime.now()
    fecha_inicial, fecha_final = asegurarse_el_formato_mysql(pregunta, fecha_actual)

    app.logger.info("Fecha inicial obtenida: %s", fecha_inicial)
    app.logger.info("Fecha final obtenida: %s", fecha_final)

    if not fecha_inicial or not fecha_final:
        app.logger.info("No se encontraron fechas válidas en la pregunta.")
        return False  # Retorna False si no se encuentran fechas válidas

    app.logger.info("ID del Chatbot utilizado: %s", chatbot_id)

    url = 'https://experimental.ciceroneweb.com/api/search-event-chatbot'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "start": fecha_inicial,
        "end": fecha_final,
        "chatbot_id": chatbot_id
    }

    try:
        app.logger.info("Enviando solicitud HTTP a: %s", url)
        app.logger.info("Payload de la solicitud: %s", json.dumps(payload))
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        eventos_data = response.json()

        # Verifica si la respuesta contiene un error
        if 'error' in eventos_data:
            app.logger.error("Error en la respuesta de la API: %s", eventos_data['error'])
            return False

        eventos_string = json.dumps(eventos_data.get('events', []))
        eventos_string = eventos_string.replace('\xa0', ' ').encode('utf-8', 'ignore').decode('utf-8')
        eventos_string = eventos_string.replace('"', '').replace('\\', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace(',', '. ')

        app.logger.info(eventos_string)
        return eventos_string

    except requests.exceptions.RequestException as e:
        app.logger.error("Error en la solicitud HTTP: %s", e)
        return False


