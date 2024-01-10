import time
import requests
import openai

openai.api_key = "OPENAI_API_KEY"  # Reemplaza con tu API Key de OpenAI.
TELEGRAM_TOKEN = "TELEGRAM_TOKEN"  # Reemplaza con tu token de Telegram.


def get_updates(offset):
  """
  token: str [Telegram bot token]
  """
  url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates'
  params = {
      'offset': offset,
      'timeout': 100,
  }
  response = requests.get(url, params=params)
  return response.json()['result']


def send_messages(chat_id, text):
  url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'

  params = {'chat_id': chat_id, 'text': text}
  response = requests.post(url, params=params)

  return response


def get_openai_response(promt):
  model_engine = 'davinci:ft-personal-2023-08-06-05-25-19'
  response = openai.Completion.create(
      engine=model_engine,
      prompt=promt,
      max_tokens=200,
      n=1,  # Una sola respuesata
      stop=' END',
      temperature=0.5,
  )

  return response.choices[0].text.strip()


def main():
  # Inicializa una variable llamada offset con el valor 0.
  offset = 0
  # Inicia un bucle infinito.
  while True:
    # Llama a nuestra función actualizadora.
    # Llama a la función get_updates() para obtener actualizaciones de mensajes.
    updates = get_updates(offset)
    # Verifica si hay actualizaciones de mensajes disponibles.
    if updates:
      # Itera sobre cada actualización de mensaje recibida.
      for update in updates:
        # Actualiza el valor de offset para evitar procesar la misma actualización nuevamente.
        offset = update['update_id'] + 1
        # Obtiene el identificador de chat de la actualización de mensaje actual.
        chat_id = update["message"]["chat"]["id"]
        # Obtiene el texto del mensaje enviado por el usuario.
        user_message = update["message"]["text"]
        # Imprime en la consola el mensaje recibido del usuario.
        print(f"Received message: {user_message}")
        # Llama a la función get_openai_response() para obtener una respuesta de OpenAI.
        GPT = get_openai_response(user_message)
        # Llama a la función send_message() para enviar la respuesta al usuario.
        send_messages(chat_id, GPT)

    else:
      # Pausa la ejecución del programa durante 1 segundo.
      time.sleep(1)


if __name__ == '__main__':
  main()  # Llama a la función main() para iniciar la ejecución del programa.
