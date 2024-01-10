import logging
import os

import openai


class ChatGptService():

    @classmethod
    def chatgpt_request(cls, capacities, temp):

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=generate_tech_competencies_textual(capacities),
            temperature=temp,
        )
        logging.info(f"response")
        logging.info(f"response {response}")
        return response.choices[0].message.content
    
    @classmethod
    def chatgpt_recommendation_request(cls, capacities, temp):

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=generate_recommendations(capacities),
            max_tokens = 500,
            temperature=temp
        )
        logging.info(f"response")
        logging.info(f"response {response}")
        return response.choices[0].message.content


def generate_tech_competencies_textual(info):

    return [
        {"role": "system", "content": "Eres un redactor que recibe una lista de requisitos para una posicion laboral y por cada requisito redacta una versión más detallada de este requisito, únicamente un requisito en cada línea, separados por un salto de línea, sin agregar una presentación a la lista."},
        {"role": "user", "content": "Proeficiencia en React, Conocimientos de seguridad en la nube, Habilidad en técnicas de recuperación de datos"},
        {"role": "assistant", "content": "- Programador con amplias habilidades en React\n, - Debe tener un alto dominio de temas de seguridad en la nube\n, -Gran capacidad para recuperar datos\n"},
        {"role": "user", "content": info}
    ]

def generate_recommendations(info):
    return [
        {"role": "system", "content": "Eres un evaluador de personal de TI que recibe una lista de competencias, y de estas competencias debes escribir una recomendacion para mejorar tal competencia, una por linea, separado por un salto de linea, sin agregar uan presentacion a la lista"},
        {"role": "user", "content": "Proeficiencia en React, Conocimientos de seguridad en la nube, Habilidad en técnicas de recuperación de datos"},
        {"role": "assistant", "content": "- Aprender a usar Hooks de React\n, - Debe tener un alto dominio de temas de seguridad en la nube\n, -Aprender tecnicas de recuperacion de datos\n"},
        {"role": "user", "content": info}
    ]