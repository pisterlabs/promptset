#!/usr/bin/env python

import os
import openai
from colorama import Fore, Style


# Configura tu clave de API de OpenAI
openai.api_key = "TU_API_KEY_AQUI"

def obtener_respuesta(pregunta):
    # Manejo de los errores de autenticación
    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are a programming solve chatbot."},
                {"role": "user", "content": pregunta}
            ]
        )
        return respuesta.choices[0].message['content']
    except openai.error.AuthenticationError:
        mensaje_error = (Fore.RED + "No se ha podido autenticar con OpenAI. Por favor, comprueba tu clave de API.\n"
                         "https://platform.openai.com/account/api-keys" + Style.RESET_ALL)
        print(mensaje_error)
        exit()

def main():
    os.system('clear' if os.name == 'posix' else 'cls')    # Mensaje de bienvenida
    print(Fore.LIGHTGREEN_EX + " Bienvenido a Gepeto, un chatbot creado con GPT-3." + Style.RESET_ALL)
    print(Fore.YELLOW + " Escribe 'salir' para terminar el chat." + Style.RESET_ALL)
    # Bucle para mantener el chat abierto
    while True:
        entrada = input(Fore.BLUE + "Tú: " + Style.RESET_ALL + " ")
        if entrada.lower() == "salir":
            print(Fore.RED + "Chat finalizado." + Style.RESET_ALL)
            break
        respuesta = obtener_respuesta(entrada)
        print(Fore.YELLOW + "OpenAI:", respuesta + Style.RESET_ALL)

if __name__ == "__main__":
    main()