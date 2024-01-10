import openai
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
import os


# Configuring your API key
openai.api_key = "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"


#Configuring your access token for the Slack API
slack_token = "xoxb-5674525547136-5650777384163-ndPsdX4Gk7HBy5TQbbuYvTDa"
app = App(token=slack_token)
"""
app = App(token=slack_token)

# Función para procesar los mensajes y generar respuestas con OpenAI
def procesar_mensaje(event, say):
    # Obtén el texto del mensaje recibido
    mensaje_texto = event['text']
    print(mensaje_texto)

    # Procesa el mensaje utilizando la API de OpenAI y genera la respuesta
    respuesta_generada = openai.Completion.create(
        engine="text-davinci-002",  # Motor de lenguaje de OpenAI
        prompt=mensaje_texto,
        max_tokens=100  # Número máximo de tokens a generar en la respuesta
    )

    # Envía la respuesta generada a Slack
    respuesta_texto = respuesta_generada['choices'][0]['text']
    say(respuesta_texto)
    print(respuesta_texto)

# Escucha los mensajes entrantes en Slack y procesa cada mensaje
@app.message(".*")  # Expresión regular para capturar todos los mensajes
def handle_message(event, say):
    procesar_mensaje(event, say)

# Inicia la aplicación de Slack
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
"""

# Función para procesar los mensajes y generar respuestas con OpenAI
def procesar_mensaje(event, say):
    # Obtén el texto del mensaje recibido
    mensaje_texto = event['text']

    # Procesa el mensaje utilizando la API de OpenAI y genera la respuesta
    respuesta_generada = openai.Completion.create(
        engine="text-davinci-002",  # Motor de lenguaje de OpenAI
        prompt=mensaje_texto,
        max_tokens=100  # Número máximo de tokens a generar en la respuesta
    )

    # Envía la respuesta generada a Slack
    respuesta_texto = respuesta_generada['choices'][0]['text']
    say(respuesta_texto)

    # Muestra la respuesta también en la terminal local
    print("Mensaje del usuario: ", mensaje_texto)
    print("Respuesta del bot: ", respuesta_texto)

# Escucha los mensajes entrantes en Slack y procesa cada mensaje
@app.message(".*")  # Expresión regular para capturar todos los mensajes
def handle_message(event, say):
    procesar_mensaje(event, say)

# Ejecuta el bot en tu máquina local
if __name__ == "__main__":
    print("Iniciando el bot en la terminal local...")
    app.start(port=3000)