# -*- coding: utf-8 -*-

import speech_recognition as sr
from termcolor import colored
import openai
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Definir la plantilla para la generación de respuestas
template = """
Tom es un gran modelo lingüístico entrenado por OpenAI.
Tom tiene una personalidad amigable y es muy útil, le encanta ayudar a los usuarios a realizar tareas en su computadora.
Tom está diseñado para ayudar en una amplia gama de tareas, desde responder a preguntas sencillas hasta proporcionar explicaciones detalladas y debates sobre una gran variedad de temas. Como modelo lingüístico, Assistant es capaz de generar textos similares a los humanos a partir de la información que recibe, lo que le permite entablar conversaciones naturales y ofrecer respuestas coherentes y relevantes para el tema en cuestión.
En general, Tom es una potente herramienta que puede ayudarte con una gran variedad de tareas y proporcionarte valiosos conocimientos e información sobre una amplia gama de temas. Tanto si necesitas ayuda con una pregunta concreta como si sólo quieres mantener una conversación sobre un tema en particular, Tom está aquí para ayudarte.
{history}
Human: {input_text}
Tom:
"""
prompt = PromptTemplate(input_variables=["input_text", "history"], template=template)

# Definir la memoria de conversación
memory = ConversationBufferMemory(memory_key="chat_history")

# Crear una instancia de OpenAI
llm = OpenAI(temperature=0.5)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Definir función de búsqueda personalizada
def search(query):
    # Aquí puedes implementar tu propia lógica de búsqueda
    return "Resultados de búsqueda para: " + query

# Definir lista de herramientas
tools = [
    Tool(
        name="Current Search",
        func=search,
        description="útil para responder a preguntas sobre la actualidad o el estado actual del mundo"
    ),
]

# Inicializar agente con herramientas y configuración
zero_shot_agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
    memory=memory,
    verbose=True,
    prompt=prompt
)

# Función para procesar la entrada del usuario
def process_input(input_text):
    if input_text.lower() == "salir":
        return None
    else:
        response = generate_response(input_text)
        return response

# Función para generar respuesta utilizando el agente
def generate_response(input_text):
    response = zero_shot_agent(input_text)
    respuesta = response['output']
    return respuesta

# Función para leer la entrada del usuario desde el teclado
def read_input():
    while True:
        input_text = input(colored("Escribe algo: ", "green"))
        response = process_input(input_text)
        if response == False:
            break
        else:
            print(colored("Tom:", "yellow"), response)

# Función para escuchar entrada de voz y generar respuesta
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(colored("Escuchando...", "green"))
        audio = r.listen(source)
        try:
            input_text = r.recognize_google(audio, language="es-ES")
            print(colored("Entrada de voz:", "yellow"), input_text)
            response = process_input(input_text)
            if response == None:
                return False
            else:
                print(colored("Tom:", "yellow"), response)
        except:
            print(colored("No se pudo reconocer la entrada de voz.", "red"))
            return True
    return True

# Función principal del programa
def main():
    print(colored("¡Buenas! Soy Tom, ¿En qué puedo ayudarte?", "magenta"))
    while True:
        print(colored("¿Cómo te gustaría interactuar?", "blue"))
        print(colored("1. Teclado", "cyan"))
        print(colored("2. Salir", "red"))
        option = input(colored("Elige una opción: ", "green"))
        if option == "3":
            should_continue = listen()
            if not should_continue:
                break
        elif option == "1":
            read_input()
        elif option == "2":
            print(colored("¡Hasta luego!", "magenta"))
            break
        else:
            print(colored("Opción inválida. Por favor, elige una opción válida.", "red"))

if __name__ == "__main__":
    main()
