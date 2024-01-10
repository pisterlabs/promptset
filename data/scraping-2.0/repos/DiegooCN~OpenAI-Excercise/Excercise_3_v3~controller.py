import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tries = 0

def function_handler(messages, function, user_prompt):
    """
    Retorna un mensaje especifico
    
    > say_hello \
    > method_payment \
    > payment_places \
    > say_goodbye \
    > get_debt_detail \
    > out_of_context \

    """

    if function == "say_hello":
        prompt = say_hello()
    elif function == "get_debt_detail":
        dni = json.loads(get_dni_from_user_prompt(messages))["dni"]
        print(dni)
        if is_dni_valid(dni):
            prompt = get_debt_detail(dni)
        else:
            prompt = ask_dni()
    elif function == "method_payment" or function == "payment_places":
        prompt = get_method_payment_locations()
    elif function == "say_goodbye":
        prompt = say_goodbye()
    elif function == "get_receipt":
        prompt = get_receipt()
    else:
        prompt = out_of_context()
    return prompt

# Ended functions
def say_hello():
    prompt = f"""¡Hola! Bienvenid@ al chat de Movistar!\nEstoy para ayudare en:\n• Conocer detalle de tu deuda\n• Formas y lugares de pago\n• Solicitar Recibo\nComentanos, ¿Qué necesitas?"""
    return prompt

def out_of_context():
    prompt = f"""Lo siento, no puedo responder a eso."""
    return prompt

def ask_dni():
    prompt = f"""Necesito consultar algunos datos para continuar con tu consulta. Por favor, ingresa el documento de identidad del titular del servicio."""
    return prompt

def get_method_payment_locations():

    """Muestra las formas y lugares de pago"""

    prompt = """\nFORMAS Y LUGARES DE PAGO\nEn Movistar te brindamos diversas formas de pago SIN COMISIÓN.\nPuedes pagar por Yape https://innovacxion.page.link/mVFa\ndesde la web o app de tu banco.\nConoce todos los canales de pago en el siguiente link\nhttps://www.movistar.com.pe/atencion-al-cliente/lugares-y-medios-de-pago"""
    return prompt   

def get_debt_detail(dni):

    """Muestra el detalle de la deuda"""

    prompt = f"""\nDETALLE DE DEUDA\nTu deuda al día de hoy es de S/ 10.00\nTu fecha de vencimiento es el 12/07/2023\nTu DNI: {dni}"""
    return prompt

def get_receipt():

    """Muestra el link para solicitar el recibo"""

    prompt = """\nSOLICITAR RECIBO\nObten tu recibo con solo unos clics\nhttps://mirecibo.movistar.com.pe"""
    return prompt

def say_goodbye():

    """Se despide del usuario cuando este lo solicite"""

    prompt = """\nGracias por usar el servicio de asistencia de Movistar\n¡Hasta pronto!"""

    return prompt


def get_dni_from_user_prompt(user_prompt):


    behavior = f"""\
    Tu objetivo es analizar el siguiente prompt {user_prompt} e identificar el DNI del usuario.\
    Luego deberás retornar un json con el siguiente formato:\
    {{"dni": "dni del usuario"}}\
    Si el usuario no ingresa un DNI este será "0"
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": behavior}],
    )

    result = response.choices[0].message.content

    return result
    
def is_dni_valid(dni):
    
    dni_with_debts = ["123456789", "205314385"]
    flag = True if dni in dni_with_debts else False
    return flag