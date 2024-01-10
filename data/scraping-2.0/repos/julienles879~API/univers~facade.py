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

#Observateur qui gére l'erreur pendant la génération de l'image d'un univers
class ImageGenerationObserver:

    # Vue qui permet d'afficher un message d'erreur dans le cas ou une image est bien généré
    def image_generation_success(self, imagePathUrl):
        print('imagePathUrl :', imagePathUrl)
        print(f"Image générée avec succès : {imagePathUrl}")

    #Vue qui permet d'afficher un message d'erreur  dans le cas ou une image est pas généré
    def image_generation_failure(self, error_response):
        print(f"Échec de la génération d'image : {error_response}")


# Facade qui permet de gérer la création d'un univers
class UniversFacade:
    observers = [
        ImageGenerationObserver()
    ]

    #Vue qui permet de gérer la création d'un univers
    @staticmethod
    def create_univers(request, name):
        try:
            utilisateur_id, username = validate_jwt_token(request.META.get('HTTP_AUTHORIZATION', '').split(' ')[1])

            if utilisateur_id is not None:
                if UniversFacade.univers_exists(name, utilisateur_id):
                    error_response = {
                        'error': f"L'univers '{name}' a déjà été créé par cet utilisateur."
                    }
                    return Response(error_response, status=status.HTTP_400_BAD_REQUEST)
                description = UniversFacade.generate_univers_description(name)
                summary = UniversFacade.generate_summary(name, description)
                imagePathUrl = UniversFacade.generate_and_save_image(name, summary)

                UniversFacade.image_generation_success(imagePathUrl)

                UniversFacade.save_univers_to_database(name, description, imagePathUrl, utilisateur_id)

                response_data = {
                    'message': 'Univers créé avec succès',
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
            UniversFacade.image_generation_failure(str(e))
            error_response = {
                'error': str(e)
            }
            return Response(error_response, status=status.HTTP_400_BAD_REQUEST)


    # Vue qui d'ajouter un observateur
    @classmethod
    def add_observer(cls, observer):
        cls.observers.append(observer)

    # Observateur qui permet de observer si une image est bien généré
    @classmethod
    def image_generation_success(cls, imagePathUrl):
        for observer in cls.observers:
            observer.image_generation_success(imagePathUrl)

    # Observateur qui permet de observer si une image n'est généré pas 
    @classmethod
    def image_generation_failure(cls, error_message):
        for observer in cls.observers:
            observer.image_generation_failure(error_message)

    # Vue qui permet de généré une description d'un univers avec chat gpt
    @staticmethod
    def generate_univers_description(name):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "assistant", "content": "You are an expert in creating descriptions for fictional universe."},
                {"role": "user", "content": f"Generate a description for the universe {name},  less 500 characters."}
            ]
        )
        description = response.choices[0].message.content
        return description

    # Vue qui permet de généré un résumé de la description d'un univers avec chat gpt
    @staticmethod
    def generate_summary(name, description):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant", "content": "You are an expert with extensive knowledge about the {name} universe"},
                {"role": "user", "content": f"Generate a summary of the description {description} to create a Text-to-Image prompt for the {name} universe in under 200 characters."}
            ],
        )
        summary = response.choices[0].message.content
        return summary


    # Vue qui permet d'ajouter un univers en bdd
    @staticmethod
    def save_univers_to_database(name, description, imagePathUrl, utilisateur_id):
        try:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO univers (name, description, imagePathUrl, id_utilisateur) VALUES (%s, %s, %s, %s)",
                               [name, description, imagePathUrl, utilisateur_id])
        except Exception as e:
            raise e
        

    # Vue qui permet de généré une image à l'aide du résumé de la descritpion grâce à stable diffusion
    @staticmethod
    def generate_and_save_image(name, summary):
        try:
            prompt = f"Generate an image of the univers {name} with the following summary: {summary}"
            r = requests.post('https://clipdrop-api.co/text-to-image/v1',
                              files={'prompt': (None, prompt, 'text/plain')},
                              headers={'x-api-key': clipdrop_api_key}
            )
            if r.ok:
                image_data = r.content
                imagePathUrl = f"media/img/univers/{name}.png"
                with open(imagePathUrl, 'wb') as image_file:
                    image_file.write(image_data)
                return imagePathUrl
            else:
                raise Exception(f'Erreur lors de la génération de l\'image: {r.status_code} - {r.text}')
        except Exception as e:
            raise e
        
    #Vue qui permet de vérifier l'existance d'un univers en bdd
    @staticmethod
    def univers_exists(name, utilisateur_id):
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM univers WHERE name = %s AND id_utilisateur = %s", [name, utilisateur_id])
            count = cursor.fetchone()[0]
            return count > 0