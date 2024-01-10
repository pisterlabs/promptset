# https://platform.openai.com/docs/guides/function-calling

# Importar la librería dotenv para cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from datetime import datetime

client = OpenAI() # Crear una instancia del cliente de OpenAI

# Crear una función asíncrona para obtener la hora actual
def get_current_time():
    date = datetime.now()
    hours = date.hour
    minutes = date.minute
    seconds = date.second
    time_of_day = "AM"
    if hours > 12:
        hours = hours - 12
        time_of_day = "PM"
    return f"{hours}:{minutes}:{seconds} {time_of_day}"

# # time = get_current_time()
# print(time)

# Crear una función asíncrona para enviar un mensaje
def send_message(message):
    functions = [
        {
            "name": "getCurrentTime",
            "description": "Get the current time of the day",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": "Eres un asistente de IA con acceso a funciones en el ordenador de los usuarios",
        },
        {
            "role": "assistant",
            "content": message,
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.9,
        messages=messages,
        functions=functions,
    )

    print(completion)
    print(completion.choices[0].message)

    """
    Aquí se podría crear un switch para ejecutar la función correspondiente
    Después del if completion.choices[0].message.function_call hacer diferentes cases
    """

    # Si la respuesta contiene una función llamada getCurrentTime
    if completion.choices[0].message.function_call and \
    completion.choices[0].message.function_call.name == "getCurrentTime":
        messages.append(completion.choices[0].message)

        current_time = get_current_time()
        # print(f"La hora actual es: {current_time}")

        messages.append({
            "role": "function", # function|system|assistant|user
            "name": "getCurrentTime",
            "content": current_time,
        })

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.9,
            messages=messages,
            functions=functions,
        )
        print(completion)
        print(completion.choices[0].message)

# Ejecutar la función send_message
send_message("Hola, ¿qué hora es?")
