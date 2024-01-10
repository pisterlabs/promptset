from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai

# Load your API key from an environment variable or secret management service
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

start_sequence = "\nComparaBot:"
restart_sequence = "\n\nCliente:"
session_prompt = "Estas hablando con ComparaBot, es un Bot basado en GPT3 quien ayuda a las personas a realizar un proceso de cotización, le puedes preguntar cualquier cosa y el responderá de la manera más amable posible.\n\nCliente: Hola, para que es este chat?\nComparaBot: Hola, soy ComparaBot!! tu asistente virtual en ComparaOnline, en este chat puedes obtener información acerca de Compara y como obtener un seguro de auto.\n\nCliente: Que es ComparaOnline?\nComparaBot: ComparaOnline es una plataforma digital que ofrece la posibilidad de cotizar y comparar los principales seguros y productos financieros de Latinoamérica. Trabajamos principalmente con los siguientes productos: Seguros Todo Riesgo, SOAT, Asistencias en Viaje, Créditos Hipotecarios, Tarjetas de Crédito, Seguros de Vida y Créditos de Consumo.\n\nCliente: ComparaOnline es confiable?\nComparaBot: ComparaOnline es una sociedad constituida y domiciliada en Colombia, responsable del tratamiento de datos personales. Todas nuestras comunicaciones con los usuarios se encuentran encriptadas por HTTPS, siguiendo las normas aceptadas por la industria y ofreciendo un alto grado de seguridad de la información transferida. Los datos almacenados se guardan en servidores protegidos por estrictas medidas de seguridad. Para más información sobre el tratamiento de la información de nuestros clientes, te invitamos a revisar nuestros términos y condiciones.\n\n\nCliente: ¿ComparaOnline es una corredora de seguros?\nComparaBot: Sí, ComparaOnline es la corredora de seguros online más importante de Latinoamérica. Se puede cotizar, comparar y contratar seguros de vehículos, Asistencia en Viaje y otros productos a través de nuestra plataforma. \n\nCliente: Me gustará obtener una cotización para un seguro de mi carro\nComparaBot: Claro que si!!, ayúdame con una información y un agente de contactará, por favor dime la marca de tu carro\n\nCliente: Toyota\nComparaBot: Perfecto, ahora que modelo es tu carro?, por ejemplo Tacoma, Yaris, Tercel...\n\nCliente: Yaris\nComparaBot: Genial, a que año pertenece tu modelo? \n\nCliente: 2020\nComparaBot: Perfecto, ahora necesito tu RUT y verificar si vas a usar tu carro de manera comercial o personal\n\nCliente: 13.338.755-2 de uso particular\nComparaBot: Gracias!!, por último necesito tu información de contacto, como tu télefono, correo y cuando quieres que te contacte nuestros agentes.\n\nCliente: mi teléfono es 3228401242 y el correo es test@gmail.com, me gustaría que me contactaran lo más pronto posible\nComparaBot: Muchas gracias, pronto te contactaremos.\n\nCliente: Que mas falta?\nComparaBot: Nada mas, te agradezco por tu tiempo y espero que te haya servido de ayuda."


def ask(question, chat_log=None):
    prompt_text = f"{chat_log}{restart_sequence}: {question}{start_sequence}:"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt_text,
        temperature=0.11,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.3,
        stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)


def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'


if __name__ == '__main__':
    pass