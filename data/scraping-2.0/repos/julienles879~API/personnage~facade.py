from rest_framework.response import Response
from rest_framework import status
from jwt.exceptions import DecodeError, ExpiredSignatureError
from django.db import connection

from api.utils import *

import openai
import os
import json 
import environ 
import requests

env = environ.Env()
environ.Env.read_env()
config = Config('/api/.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

clipdrop_api_key= os.getenv("SD_API_KEY")


# Facade qui permet de gérer la création d'un personnage
class PersonnageFacade:
    
    # Vue qui permet de créer un personnage.
    # Cette vue appelle aussi les vue de génération de description, résumé et d'image.
    @staticmethod
    def create_personnage(request, univers_id, name):
        try:
            utilisateur_id, username = validate_jwt_token(request.META.get('HTTP_AUTHORIZATION', '').split(' ')[1])

            if utilisateur_id is not None:
                data = json.loads(request.body.decode('utf-8'))
                name = data.get('name')

                description = PersonnageFacade.generate_character_description(name)
                summary = PersonnageFacade.generate_summary(name, description)
                imagePathUrl = PersonnageFacade.generate_and_save_image(name, summary)

                PersonnageFacade.save_personnage_to_database(name, description, imagePathUrl, univers_id)

                response_data = {
                    'message': 'Personnage créé avec succès',
                    'description': description,
                    'summary': summary,
                    'imagePathUrl': imagePathUrl
                }
                return Response(response_data, status=status.HTTP_201_CREATED)
            else:
                error_response = {
                    'error': 'Token invalide'
                }
                return Response(error_response, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            error_response = {
                'error': str(e)
            }
            return Response(error_response, status=status.HTTP_400_BAD_REQUEST)

     # Vue qui permet de générer une description grace a chat gpt
    @staticmethod
    def generate_character_description(character_name):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "assistant", "content": "You are an expert in creating character descriptions."},
                {"role": "user", "content": f"Generate a description for the character {character_name}"}
            ]
        )
        description = response.choices[0].message.content

        return description


     # Vue qui permet de générer un résumé grace à la description grace a chat gpt
    @staticmethod
    def generate_summary(name, description):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "assistant", "content": "You are an expert in generating character descriptions and summaries."},
                {"role": "user", "content": f"Generate a summary of the description {description} to create a Text-to-Image prompt for the {name} universe in under 200 characters."}
            ]
        )
        summary = response.choices[0].message.content

        return summary


    # Vue qui permet de générer une image à l'aide du résumé grace à stable diffusion
    @staticmethod
    def generate_and_save_image(name, summary):
        try:
            prompt = f"Generate an image of the character {name} with the following summary: {summary}"

            r = requests.post('https://clipdrop-api.co/text-to-image/v1',
                              files={'prompt': (None, prompt, 'text/plain')},
                              headers={'x-api-key': clipdrop_api_key}
            )

            if r.ok:
                image_data = r.content

                image_path = f"media/img/personnages/{name}.png"
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_data)

                return image_path
            else:
                raise Exception(f'Erreur lors de la génération de l\'image: {r.status_code} - {r.text}')
        except Exception as e:
            raise e

    # Vue qui permet d'ajouter un personnage en bdd
    @staticmethod
    def save_personnage_to_database(name, description, imagePathUrl, univers_id):
        try:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO personnage (name, description, imagePathUrl, id_univers) VALUES (%s, %s, %s, %s)",
                               [name, description, imagePathUrl, univers_id])

        except Exception as e:
            raise e